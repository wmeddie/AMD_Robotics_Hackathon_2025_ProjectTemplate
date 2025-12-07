# Plan: Integrating MCP Tools into Xircuits Agent Components (Declarative Approach)

This plan outlines the steps to integrate Model Context Protocol (MCP) tools into the Xircuits agent components, leveraging the central MCP configuration managed by the core system.

## Core Idea

Instead of Xircuits components directly managing MCP connections, they will declare which *pre-configured* MCP servers (defined in `mcp_settings.json` or equivalent) should be active for a given agent run. The agent runtime (`AgentRun` and supporting components) will then interact with the core system to discover and invoke tools from these declared servers.

## Steps

1.  **New Component: `AgentUseMCPTools`**
    *   **Purpose:** Declares that tools from a specific, already configured MCP server should be available.
    *   **Inputs:**
        *   `server_name`: `InCompArg[str]` - The exact name of the MCP server from the configuration file.
        *   `toolbelt_name`: `InArg[str]` (Optional, default: 'default') - Conceptual grouping name for `ctx`.
    *   **Outputs:** None directly.
    *   **Implementation (`execute`):** Adds the `server_name` to a list in the context: `ctx.setdefault('mcp_servers_' + toolbelt_name, []).append(self.server_name.value)`.

2.  **Modify `AgentMakeToolbelt`**
    *   **Purpose:** Collects standard tool definitions and declared MCP server names to create the final specification for `AgentInit`.
    *   **Inputs/Outputs:** Remain the same.
    *   **Implementation (`execute`):**
        *   Initialize `standard_tools = {}` and `mcp_servers = []`.
        *   Process standard tools from `ctx['toolbelt_' + toolbelt_name]` into `standard_tools`.
        *   Retrieve MCP server names from `ctx['mcp_servers_' + toolbelt_name]` into `mcp_servers`.
        *   Create the output dictionary: `self.toolbelt_spec.value = {'standard_tools': standard_tools, 'mcp_servers': mcp_servers}`.

3.  **Modify `AgentInit`**
    *   **Purpose:** Initialize the agent context with standard tools and enabled MCP server names.
    *   **Inputs:** Accepts the `toolbelt_spec` dictionary.
    *   **Implementation (`execute`):** Store both parts in the agent's context:
        ```python
        agent_context = ctx['agent_' + self.agent_name.value]
        agent_context['agent_toolbelt'] = self.toolbelt_spec.value.get('standard_tools', {})
        agent_context['enabled_mcp_servers'] = self.toolbelt_spec.value.get('mcp_servers', [])
        # Store other existing agent properties...
        ```

4.  **Modify `AgentRun`**
    *   **Tool Prompt Generation (`make_tools_prompt`):**
        *   Needs access to `agent_toolbelt` and `enabled_mcp_servers`.
        *   List standard tools.
        *   For each server in `enabled_mcp_servers`, query the *core system* for its tool list (name, description) and add them to the prompt.
    *   **Tool Execution (`handle_tool_use`):**
        *   If `TOOL: <tool_name> <args>` is called:
        *   Check if `<tool_name>` is a standard tool in `agent_toolbelt`. If yes, execute it.
        *   If not, query the *core system* to see if `<tool_name>` belongs to any server in `enabled_mcp_servers`.
        *   If yes (for `matched_server_name`), request the *framework* to execute `use_mcp_tool` for that server/tool/args.
        *   Append the result returned by the framework to the conversation.

## Visualization

```mermaid
graph TD
    subgraph Xircuits Setup
        UserDefinesTool --> ADT(AgentDefineTool)
        ADT -- Adds self instance to --> CtxToolbeltCollection{ctx['toolbelt_...']}

        UserInput_ServerName[User Input: server_name] --> AUMCP(AgentUseMCPTools)
        AUMCP -- Adds server_name to --> CtxMCPList{ctx['mcp_servers_...']}

        AMT(AgentMakeToolbelt) -- Reads --> CtxToolbeltCollection
        AMT -- Reads --> CtxMCPList
        AMT -- Creates --> ToolSpecDict{toolbelt_spec: {standard_tools: {...}, mcp_servers: [...]}}
        AMT -- Outputs --> toolbelt_spec_out([toolbelt_spec])
        toolbelt_spec_out --> AI(AgentInit)
    end

    subgraph Agent Runtime Initialization
         AI -- Stores in agent context --> AgentContext{Agent Context: standard_tools, enabled_mcp_servers, ...}
    end

    subgraph AgentRun Execution
        AR(AgentRun) -- Uses --> AgentContext

        subgraph Tool Prompt Generation
            AR -- Calls --> MakePrompt(make_tools_prompt)
            MakePrompt -- Gets standard tools --> AgentContext
            MakePrompt -- Gets enabled MCP server names --> AgentContext
            MakePrompt -- For each enabled server, queries --> CoreSystem(Core System MCP Knowledge)
            CoreSystem -- Returns tool list for server --> MakePrompt
            MakePrompt -- Generates --> SystemPromptText(System Prompt w/ All Tools)
        end

        subgraph Tool Execution Handling
            AR -- LLM calls tool --> ToolLookup{Lookup tool}
            ToolLookup -- Check Standard Tools --> StandardToolCheck{Found in standard_tools?}
            StandardToolCheck -- Yes --> ExecuteStandardTool(Execute Standard Tool Callable)

            StandardToolCheck -- No --> MCPToolCheck{Belongs to enabled MCP Server?}
            MCPToolCheck -- Queries System --> CoreSystem
            MCPToolCheck -- Yes --> InvokeFrameworkMCP(AR requests Framework executes use_mcp_tool)
            InvokeFrameworkMCP -- Framework uses server config --> MCPComms{Framework MCP Communication}
            MCPComms -- Interacts --> MCPServer(Configured MCP Server)
            MCPComms -- Returns result to Framework --> InvokeFrameworkMCP
            InvokeFrameworkMCP -- Returns result --> AR

            MCPToolCheck -- No --> HandleError(Tool Not Found Error)
            ExecuteStandardTool -- Returns result --> AR
            AR -- Appends Result --> ConversationHistory[...]
        end
    end
```

## Assumptions

*   The core system running the Xircuits graph has loaded the MCP configuration and can provide information about configured servers and their tools upon request.
*   The core system provides a mechanism for `AgentRun` to request the execution of `use_mcp_tool` for a specific server, tool, and arguments, handling the actual MCP communication.