// extension.ts - VS Code extension for Morphic Markdown Agent
import * as vscode from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions } from 'vscode-languageclient/node';

export function activate(context: vscode.ExtensionContext) {
    // Register the LSP client for your morphic agent
    const serverOptions: ServerOptions = {
        command: 'python3',
        args: [context.asAbsolutePath('md_agent.py'), 'lsp'], // You'd need to add LSP mode
        options: {
            env: {
                ...process.env,
                PYTHONPATH: context.asAbsolutePath('morphic_modules')
            }
        }
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'markdown' },
            { scheme: 'file', pattern: '**/*.md' }
        ],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.md')
        }
    };

    const client = new LanguageClient(
        'morphicMarkdownAgent',
        'Morphic Markdown Agent',
        serverOptions,
        clientOptions
    );

    // Register custom commands for morphic operations
    const extractSectionCommand = vscode.commands.registerCommand(
        'morphic.extractSection',
        async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'markdown') {
                vscode.window.showErrorMessage('Please open a markdown file');
                return;
            }

            // Get cursor position to determine which section to extract
            const position = editor.selection.active;
            const document = editor.document;
            
            // Request code actions from LSP server
            const codeActions = await vscode.commands.executeCommand<vscode.CodeAction[]>(
                'vscode.executeCodeActionProvider',
                document.uri,
                new vscode.Range(position, position),
                vscode.CodeActionKind.Refactor
            );

            const extractActions = codeActions?.filter(action => 
                action.title.includes('Extract') && action.title.includes('section')
            );

            if (extractActions && extractActions.length > 0) {
                // Show quick pick for available sections
                const items = extractActions.map(action => ({
                    label: action.title,
                    action: action
                }));

                const selected = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select section to extract'
                });

                if (selected && selected.action.command) {
                    await vscode.commands.executeCommand(
                        selected.action.command.command,
                        ...selected.action.command.arguments || []
                    );
                }
            }
        }
    );

    const inlineSectionCommand = vscode.commands.registerCommand(
        'morphic.inlineSection',
        async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'markdown') {
                vscode.window.showErrorMessage('Please open a markdown file');
                return;
            }

            const position = editor.selection.active;
            const document = editor.document;
            
            // Get code actions for inline operations
            const codeActions = await vscode.commands.executeCommand<vscode.CodeAction[]>(
                'vscode.executeCodeActionProvider',
                document.uri,
                new vscode.Range(position, position),
                vscode.CodeActionKind.Refactor
            );

            const inlineActions = codeActions?.filter(action => 
                action.title.includes('Inline') && action.title.includes('section')
            );

            if (inlineActions && inlineActions.length > 0) {
                // Execute the inline action
                const action = inlineActions[0];
                if (action.command) {
                    await vscode.commands.executeCommand(
                        action.command.command,
                        ...action.command.arguments || []
                    );
                }
            } else {
                vscode.window.showInformationMessage('No extracted sections found at cursor position');
            }
        }
    );

    const analyzeMorphicCommand = vscode.commands.registerCommand(
        'morphic.analyzeDocument',
        async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'markdown') {
                vscode.window.showErrorMessage('Please open a markdown file');
                return;
            }

            // Execute morphic analysis
            const result = await client.sendRequest('morphic/analyze', {
                textDocument: { uri: editor.document.uri.toString() }
            });

            // Show analysis results in a new document
            const doc = await vscode.workspace.openTextDocument({
                content: JSON.stringify(result, null, 2),
                language: 'json'
            });
            await vscode.window.showTextDocument(doc);
        }
    );

    // Register hover provider for morphic IDs
    const hoverProvider = vscode.languages.registerHoverProvider('markdown', {
        provideHover(document, position) {
            const range = document.getWordRangeAtPosition(position, /[a-f0-9]{8}/);
            if (range) {
                const morphicId = document.getText(range);
                // Request morphic information from LSP
                return client.sendRequest('morphic/hover', {
                    textDocument: { uri: document.uri.toString() },
                    position: position,
                    morphicId: morphicId
                }).then((result: any) => {
                    if (result) {
                        return new vscode.Hover([
                            `**Morphic Section ID**: ${morphicId}`,
                            `**Binary**: ${result.binary}`,
                            `**Level**: ${result.level}`,
                            `**Quantum State**: ${result.quantumState || 'N/A'}`
                        ]);
                    }
                });
            }
        }
    });

    // Add status bar item showing document morphic state
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'morphic.analyzeDocument';
    statusBarItem.tooltip = 'Click to analyze morphic document structure';

    const updateStatusBar = async () => {
        const editor = vscode.window.activeTextEditor;
        if (editor && editor.document.languageId === 'markdown') {
            try {
                const analysis = await client.sendRequest('morphic/quickAnalysis', {
                    textDocument: { uri: editor.document.uri.toString() }
                });
                statusBarItem.text = `$(symbol-structure) Morphic: ${analysis.sectionCount} sections, entropy: ${analysis.entropy.toFixed(2)}`;
                statusBarItem.show();
            } catch {
                statusBarItem.hide();
            }
        } else {
            statusBarItem.hide();
        }
    };

    // Update status bar when active editor changes
    vscode.window.onDidChangeActiveTextEditor(updateStatusBar);
    vscode.workspace.onDidChangeTextDocument(e => {
        if (e.document === vscode.window.activeTextEditor?.document) {
            updateStatusBar();
        }
    });

    // Initial status bar update
    updateStatusBar();

    // Start the LSP client
    context.subscriptions.push(
        client.start(),
        extractSectionCommand,
        inlineSectionCommand,
        analyzeMorphicCommand,
        hoverProvider,
        statusBarItem
    );

    // Register keybindings in package.json:
    // "contributes": {
    //   "keybindings": [
    //     {
    //       "command": "morphic.extractSection",
    //       "key": "ctrl+shift+e",
    //       "when": "editorTextFocus && editorLangId == markdown"
    //     },
    //     {
    //       "command": "morphic.inlineSection", 
    //       "key": "ctrl+shift+i",
    //       "when": "editorTextFocus && editorLangId == markdown"
    //     },
    //     {
    //       "command": "morphic.analyzeDocument",
    //       "key": "ctrl+shift+m",
    //       "when": "editorTextFocus && editorLangId == markdown"
    //     }
    //   ]
    // }
}