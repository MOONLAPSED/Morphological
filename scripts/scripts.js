window.onload = function () {
  const replInput = document.getElementById('repl-input');
  const replOutput = document.getElementById('repl-output');
  const markdownEditor = document.getElementById('markdown-editor');
  const executeButton = document.getElementById('execute');

  executeButton.addEventListener('click', function () {
    const code = replInput.value;
    const markdown = markdownEditor.value;

    replOutput.textContent = `>>> ${code}\n\n[Markdown Preview Placeholder]\n${markdown}`;
    replInput.value = '';
  });

  markdownEditor.addEventListener('keydown', function (event) {
    if (event.key === "Tab") {
      event.preventDefault();
      const start = this.selectionStart;
      const end = this.selectionEnd;
      this.value = this.value.substring(0, start) + "\t" + this.value.substring(end);
      this.selectionStart = this.selectionEnd = start + 1;
    }
  });
}
