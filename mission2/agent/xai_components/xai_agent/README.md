<p align="center">
  <a href="https://github.com/XpressAI/xircuits/tree/master/xai_components#xircuits-component-library-list">Component Libraries</a> •
  <a href="https://github.com/XpressAI/xircuits/tree/master/project-templates#xircuits-project-templates-list">Project Templates</a>
  <br>
  <a href="https://xircuits.io/">Docs</a> •
  <a href="https://xircuits.io/docs/Installation">Install</a> •
  <a href="https://xircuits.io/docs/category/tutorials">Tutorials</a> •
  <a href="https://xircuits.io/docs/category/developer-guide">Developer Guides</a> •
  <a href="https://github.com/XpressAI/xircuits/blob/master/CONTRIBUTING.md">Contribute</a> •
  <a href="https://www.xpress.ai/blog/">Blog</a> •
  <a href="https://discord.com/invite/vgEg2ZtxCw">Discord</a>
</p>





<p align="center"><i>Xircuits Component Library for Agent! Enable intelligent agents with memory, tool usage, and dynamic decision-making.</i></p>

---

### Xircuits Component Library for Agent
This library empowers Xircuits with components to build intelligent agents that can interact dynamically with users, store and query memories, and utilize tools for specific tasks. It supports OpenAI and VertexAI agents for creating advanced conversational systems.

## Table of Contents

- [Preview](#preview)
- [Prerequisites](#prerequisites)
- [Main Xircuits Components](#main-xircuits-components)
- [Try the Examples](#try-the-examples)
- [Installation](#installation)

## Preview

### The Example:

![agent_example](https://github.com/user-attachments/assets/ee07936d-5c28-4593-b7b6-002e54b5145f)

### The Result:

![agent_example_result](https://github.com/user-attachments/assets/0156012e-3fac-4030-8f77-df3d9fb39bc4)

## Prerequisites

Before you begin, you will need the following:

1. Python3.9+.
2. Xircuits.

## Main Xircuits Components

### AgentInit Component:
Initializes the agent with a name, provider (OpenAI/VertexAI), memory, tools, and system prompts. This is the foundation of the agent setup.

<img src="https://github.com/user-attachments/assets/568723d1-d5b2-4a1f-94ac-04f7f1dbb714" alt="AgentInit" width="200" height="200" />

### AgentRun Component:
Runs the agent with a given conversation and allows the agent to use tools dynamically. Outputs the conversation history and the agent's last response.

<img src="https://github.com/user-attachments/assets/6f769f45-e7ca-4805-8f2a-5876596651ab" alt="AgentRun" width="200" height="100" />

### AgentDefineTool Component:
Defines a tool for the agent to use during interactions. The tool can perform custom actions based on user input.

### AgentToolOutput Component:
Outputs the result of a tool back to the agent, enabling it to complete its task or provide a response.

### AgentMakeToolbelt Component:
Creates a collection of tools (toolbelt) for the agent, which can be referenced in workflows to enhance functionality.

### AgentNumpyMemory Component:
Implements local, temporary memory using NumPy for storing and querying embeddings.

### AgentVectoMemory Component:
Integrates with the Vecto API to provide external memory support for long-term storage and querying.

### AgentStreamStringResponse Component:
Streams a response string word-by-word, mimicking the typing effect often seen in modern conversational systems.

## Try The Examples

We have provided an example workflow to help you get started with the Agent component library. Give it a try and see how you can create custom Agent components for your applications.

### Agent Example

This example uses the AgentRun component to create a conversational agent named HodlBot. The agent uses the get_current_time tool to respond to user queries about the current time in specific locations.

## Installation

To use this component library, ensure that you have an existing [Xircuits setup](https://xircuits.io/docs/main/Installation). You can then install the Agent library using the [component library interface](https://xircuits.io/docs/component-library/installation#installation-using-the-xircuits-library-interface), or through the CLI using:

```
xircuits install agent
```
You can also do it manually by cloning and installing it:
```
# base Xircuits directory
git clone https://github.com/XpressAI/xai-agent xai_components/xai_agent
pip install -r xai_components/xai_agent/requirements.txt

```
