import asyncio
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional
import http.client
from urllib.parse import urlparse

class LlamaInterface:
    def __init__(self):
        self.session = None
        self.mock_mode = False

    async def __aenter__(self):
        try:
            self.session = http.client.HTTPConnection("localhost", 11434)
            self.session.connect()
        except ConnectionRefusedError:
            print("Warning: Unable to connect to Llama server. Switching to mock mode.")
            self.mock_mode = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session and not self.mock_mode:
            self.session.close()

    async def _query_llama(self, prompt):
        if self.mock_mode:
            return f"Mock response for: {prompt}"

        if not self.session:
            raise RuntimeError("LlamaInterface must be used as an async context manager")

        payload = json.dumps({"model": "gemma2", "prompt": prompt, "stream": False})
        headers = {"Content-Type": "application/json"}

        try:
            self.session.request("POST", "/api/generate", body=payload, headers=headers)
            response = self.session.getresponse()

            if response.status == 200:
                result = json.loads(response.read().decode())
                return result["response"]
            else:
                raise Exception(f"API request failed with status {response.status}")
        except Exception as e:
            print(f"Error querying Llama: {e}")
            return f"Error response for: {prompt}"

    async def extract_concepts(self, text):
        prompt = f"Extract key concepts from the following text:\n\n{text}\n\nConcepts:"
        response = await self._query_llama(prompt)
        return [concept.strip() for concept in response.split(",")]

    async def process(self, task):
        return await self._query_llama(task)

class SymbolicKernel:
    def __init__(self, kb_dir, output_dir, max_memory):
        self.kb_dir = kb_dir
        self.output_dir = output_dir
        self.max_memory = max_memory
        self.llama = None
        self.running = False
        self.knowledge_base = set()

    async def __aenter__(self):
        self.llama = await LlamaInterface().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.llama:
            await self.llama.__aexit__(exc_type, exc, tb)

    async def initialize(self):
        self.llama = await LlamaInterface().__aenter__()
        self.running = True

    async def process_task(self, task):
        if not self.running:
            raise RuntimeError("Kernel is not initialized or has been stopped")
        result = await self.llama._query_llama(task)
        concepts = await self.llama.extract_concepts(result)
        self.knowledge_base.update(concepts)
        return result

    async def stop(self):
        self.running = False
        if self.llama:
            await self.llama.__aexit__(None, None, None)

    def get_status(self):
        return {"kb_size": len(self.knowledge_base), "running": self.running}

    async def query(self, query):
        if not self.running:
            raise RuntimeError("Kernel is not initialized or has been stopped")
        return await self.llama._query_llama(query)

class AdaptiveKnowledgeBase:
    def __init__(self, kb_dir: str, output_dir: str, max_memory: int):
        self.symbolic_kernel = SymbolicKernel(kb_dir, output_dir, max_memory)
        self.evolution_history = []
        self.experiments = []

    async def initialize(self):
        await self.symbolic_kernel.initialize()
        self.commit_changes("System initialized")

    async def run_experiments(self):
        for dual_experiment in self.experiments:
            results = await dual_experiment.run(self.symbolic_kernel)
            await self.evolve_based_on_results(results)
            yield results

    async def evolve_based_on_results(self, results: Dict[str, Any]):
        original_concepts = await self.symbolic_kernel.llama.extract_concepts(str(results["original"]))
        negation_concepts = await self.symbolic_kernel.llama.extract_concepts(str(results["negation"]))
        self.symbolic_kernel.knowledge_base.update(original_concepts + negation_concepts)
        self.commit_changes("Evolved based on experiment results")

    def commit_changes(self, message: str):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.evolution_history.append({"timestamp": timestamp, "message": message})

    def get_evolution_history(self) -> List[Dict[str, str]]:
        return self.evolution_history

# --- Mock Experiment Class for Demo Purposes ---
class DualExperiment:
    def __init__(self, input_prompt: str):
        self.input_prompt = input_prompt

    async def run(self, kernel: SymbolicKernel) -> Dict[str, str]:
        original = await kernel.process_task(self.input_prompt)
        negation = await kernel.process_task(f"Not: {self.input_prompt}")
        return {"original": original, "negation": negation}

# --- Main Demo Function ---
async def main():
    system = AdaptiveKnowledgeBase("kb_dir", "output_dir", 1000000)
    await system.initialize()

    # Add mock experiment(s)
    system.experiments.append(DualExperiment("What are the benefits of AI in education?"))

    print("Running experiments...")
    async for results in system.run_experiments():
        print(f"Results:\n  Original: {results['original']}\n  Negation: None")  # Negation: {results['negation']}
        print(f"Knowledge Base Size: {system.symbolic_kernel.get_status()['kb_size']}")
    
    import textwrap

    def pretty_print_response(title, text):
        print(f"\n{title}:\n" + textwrap.indent(textwrap.fill(text, width=80), prefix="  "))

if __name__ == "__main__":
    asyncio.run(main())
