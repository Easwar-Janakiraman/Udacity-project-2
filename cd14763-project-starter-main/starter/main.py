"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the project instructions and rubric for guidance.
Work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
# Create a BedrockAgentCoreApp instance.
# This registers the ASGI server for AgentCore deployment.
# There must be exactly one instance per deployment.
#
# Hint: app = BedrockAgentCoreApp()

# TODO: Create the BedrockAgentCoreApp instance
app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Replace the placeholder strings with your actual AWS resource values.
# You collected these in the infrastructure setup section of the project instructions.
#
# GATEWAY_URL format: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
# This starter uses an unsigned MCP connection and therefore assumes the
# project Gateway is configured with the NONE authorizer.
# KB_ID       format: 10-character alphanumeric string from the KB console
# REGION:     your AWS region, e.g. "us-east-1"
# MEMORY_ID   format: shown in the AgentCore Memory console

GATEWAY_URL = "https://customersupportgateway-630q2cntc0.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"   # TODO: Replace with your Gateway URL
KB_ID       = "CDAKLIHF8V"          # TODO: Replace with your Knowledge Base ID
REGION      = "us-east-1"        # TODO: Replace with your AWS region
MEMORY_ID   = "CustomerSupportMemory-22B3IX24wm"        # TODO: Replace with your Memory ID


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
# Create:
#   1. A BedrockModel using model_id "global.amazon.nova-2-lite-v1:0"
#   2. A MemoryClient with region_name=REGION
#   3. A boto3 client for the "bedrock-agent-runtime" service in REGION
#
# Hint: model = BedrockModel(model_id=model_id)

model_id = "global.amazon.nova-2-lite-v1:0"

# TODO: Create the BedrockModel instance
model = BedrockModel(model_id=model_id)

# TODO: Create the MemoryClient instance
memory_client = MemoryClient(region_name=REGION)

# TODO: Create the boto3 bedrock-agent-runtime client
_bedrock_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
# Implement get_namespaces() to return a dict mapping strategy type to
# namespace template string.
#
# Steps:
#   1. Call mem_client.get_memory_strategies(memory_id) to get strategy list
#   2. Read the namespace from strategy["namespaceTemplates"][0].
#      For compatibility with older AgentCore responses, fall back to
#      strategy["namespaces"][0] when namespaceTemplates is absent.
#
# Example output:
#   { "SEMANTIC": "cs_agent/{actorId}/facts",
#     "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)

    namespaces = {}

    for strategy in strategies:
        strategy_type = strategy.get("type")

        namespace_templates = strategy.get("namespaceTemplates")

        if not namespace_templates:
            namespace_templates = strategy.get("namespaces", [])

        if namespace_templates:
            namespaces[strategy_type] = namespace_templates[0]

    return namespaces


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
# Implement MemoryHook, a HookProvider subclass that adds long-term memory.
#
# The class needs:
#   __init__(self, actor_id, session_id, memory_client, memory_id)
#     — store all four as instance attributes
#     — call get_namespaces() and store the result as self.namespaces
#
#   retrieve_customer_context(self, event: MessageAddedEvent)
#     — only runs for plain-text user messages (not tool results)
#     — for each strategy namespace, call memory_client.retrieve_memories(
#          memory_id, namespace (formatted with actorId), query, top_k=5)
#     — collect non-empty memory texts tagged with their strategy type
#     — if any memories found, prepend them to the user message as:
#          "Customer Context:\n<memories>\n\n<original_message>"
#
#   save_support_interaction(self, event: AfterInvocationEvent)
#     — walk the message list backwards to find the last plain-text user
#       query and the last assistant response
#     — call memory_client.create_event(memory_id, actor_id, session_id,
#          messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")])
#
#   register_hooks(self, registry: HookRegistry)
#     — register retrieve_customer_context on MessageAddedEvent
#     — register save_support_interaction on AfterInvocationEvent

class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        # TODO: Store actor_id, session_id, memory_id, memory_client as attributes
        # TODO: Call get_namespaces() and store the result as self.namespaces
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages

        if not messages:
            return

        last_message = messages[-1]

        # Only process user messages.
        if last_message.get("role") != "user":
            return

        content = last_message.get("content")

        if not isinstance(content, list):
            return

        # Extract plain-text user query.
        text_parts = []

        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])

        if not text_parts:
            return

        query = "\n".join(text_parts)

        memories = []

        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.format(actorId=self.actor_id)

            try:
                results = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=query,
                    top_k=5,
                )

                for result in results:
                    memory_text = result.get("content", {}).get("text")

                    if memory_text:
                        memories.append(
                            f"[{strategy_type}] {memory_text}"
                        )

            except Exception as e:
                logger.warning(
                    "Failed to retrieve memories from %s: %s",
                    strategy_type,
                    e,
                )

        if not memories:
            return

        context = "Customer Context:\n" + "\n".join(memories)

        original_message = "\n".join(text_parts)

        new_message = (
            f"{context}\n\n"
            f"{original_message}"
        )

        last_message["content"] = [
            {"text": new_message}
        ]

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = event.agent.messages

        customer_query = None
        agent_response = None

        # Walk backwards so we get the most recent user/assistant messages.
        for message in reversed(messages):

            role = message.get("role")
            content = message.get("content")

            if not content:
                continue

            text = ""

            if isinstance(content, str):
                text = content

            elif isinstance(content, list):
                text_parts = []

                for block in content:
                    if isinstance(block, dict) and block.get("text"):
                        text_parts.append(block["text"])

                text = " ".join(text_parts)

            if not text.strip():
                continue

            if role == "assistant" and agent_response is None:
                agent_response = text

            elif role == "user" and customer_query is None:
                customer_query = text

            if customer_query and agent_response:
                break

        if not customer_query or not agent_response:
            return

        try:
            self.memory_client.create_event(
                self.memory_id,
                self.actor_id,
                self.session_id,
                messages=[
                    (customer_query, "USER"),
                    (agent_response, "ASSISTANT"),
                ],
            )

        except Exception as exc:
            logger.warning(
                "Failed to save support interaction: %s",
                exc,
            )

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(
            MessageAddedEvent,
            self.retrieve_customer_context,
        )

        registry.add_callback(
            AfterInvocationEvent,
            self.save_support_interaction,
        )


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
# Implement search_knowledge_base(query) using the @tool decorator.
#
# Steps:
#   1. Guard: if KB_ID is empty return "Knowledge base not configured."
#   2. Call _bedrock_runtime.retrieve(
#          knowledgeBaseId=KB_ID,
#          retrievalQuery={"text": query}
#      )
#   3. Extract resp["retrievalResults"]; return a message if empty
#   4. Join the text chunks with "\n---\n" and return the result
#
# The docstring is the tool description — the model uses it to decide when
# to call this tool, so keep it clear and accurate.

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."

    try:
        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )

        results = response.get("retrievalResults", [])

        if not results:
            return "No relevant information found in the knowledge base."

        chunks = []

        for result in results:
            content = result.get("content", {})
            text = content.get("text")

            if text:
                chunks.append(text)

        if not chunks:
            return "No relevant information found in the knowledge base."

        return "\n---\n".join(chunks)

    except Exception as e:
        logger.exception("Knowledge Base search failed")
        return f"Knowledge base search failed: {e}"


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
# Implement calculate_loyalty_discount() using the @tool decorator.
#
# The tool must:
#   1. Build a self-contained Python code string that:
#        • Defines earn_rates: {"standard": 1, "device": 2, "fresh": 5}
#        • Defines tier_rates: {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
#        • Calculates points_redeemed (floor to nearest 500, cap at 50% of order)
#        • Calculates tier_discount (applied to subtotal after points)
#        • Calculates final_total, total_savings, points_earned, remaining_points
#        • Prints a JSON result dict
#   2. Execute the code with code_session(REGION).invoke("executeCode", {...})
#      using language="python" and clearContext=True
#   3. Return the first result event as a JSON string
#   4. Include a fallback that computes only the tier discount if the
#      Code Interpreter is unavailable

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    # TODO: Build the code string (use an f-string to inject the arguments)
    code = f"""
import json
import math

loyalty_points = {loyalty_points}
tier = {tier!r}
order_total = {order_total}
product_category = {product_category!r}

earn_rates = {{
    "standard": 1,
    "device": 2,
    "fresh": 5
}}

tier_rates = {{
    "Silver": 0.00,
    "Gold": 0.10,
    "Platinum": 0.15
}}

earn_rate = earn_rates.get(product_category, 1)
tier_rate = tier_rates.get(tier, 0.00)

points_redeemed = min(
    (loyalty_points // 500) * 500,
    math.floor(order_total * 0.50 / 0.01)
)

points_value = points_redeemed / 100

subtotal_after_points = max(
    order_total - points_value,
    0
)

tier_discount = subtotal_after_points * tier_rate

final_total = max(
    subtotal_after_points - tier_discount,
    0
)

total_savings = order_total - final_total

points_earned = math.floor(final_total * earn_rate)

remaining_points = loyalty_points - points_redeemed

result = {{
    "points_redeemed": points_redeemed,
    "points_value": round(points_value, 2),
    "tier_discount": round(tier_discount, 2),
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points
}}

print(json.dumps(result))
"""

    try:
        with code_session(REGION) as session:
            response = session.invoke(
                "executeCode",
                {
                    "language": "python",
                    "code": code,
                    "clearContext": True,
                },
            )

        if not response:
            raise RuntimeError("Code Interpreter returned no result")

        first_result = response[0] if isinstance(response, list) else response

        if isinstance(first_result, str):
            return first_result

        return json.dumps(first_result)

    except Exception as e:
        logger.warning(
            "Code Interpreter unavailable, using fallback: %s",
            e,
        )

        tier_rates = {
            "Silver": 0.00,
            "Gold": 0.10,
            "Platinum": 0.15,
        }

        tier_rate = tier_rates.get(tier, 0.00)
        tier_discount = order_total * tier_rate
        final_total = order_total - tier_discount

        return json.dumps({
            "points_redeemed": 0,
            "tier_discount": round(tier_discount, 2),
            "final_total": round(final_total, 2),
            "total_savings": round(tier_discount, 2),
            "points_earned": 0,
            "remaining_points": loyalty_points,
            "fallback": True,
        })


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
# Implement the invoke() function decorated with @app.entrypoint.
#
# Steps:
#   1. Extract user_input, actor_id, and session_id from the payload
#      (generate a UUID if session_id is missing)
#   2. Instantiate MemoryHook for this actor/session
#   3. Instantiate AgentCoreBrowser(region=REGION)
#   4. Build the tools list: [search_knowledge_base, calculate_loyalty_discount,
#                              agent_core_browser.browser]
#   5. Connect to the Gateway via MCPClient, load gateway_tools, extend tools list
#   6. Create and invoke the Agent with all tools, hooks, and system_prompt
#   7. Return the text from the first content block of the response
#   8. Handle exceptions gracefully

@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        # 1. Extract input values
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "anonymous")
        session_id = payload.get("session_id") or str(uuid.uuid4())

        if not user_input:
            return "Please provide a prompt."

        # 2. Create the memory hook
        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        # 3. Create AgentCore Browser
        agent_core_browser = AgentCoreBrowser(region=REGION)

        # 4. Build the local tools
        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        # 5. Connect to AgentCore Gateway
        gateway_client = MCPClient(
            lambda: streamable_http_client(GATEWAY_URL)
        )

        with gateway_client:
            gateway_tools = gateway_client.list_tools_sync()

            # Add Gateway tools to the local tools
            tools.extend(gateway_tools)

            # 6. Create the Strands Agent
            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt="""
You are a helpful customer support assistant for an e-commerce platform.

You can help customers with:
- Order tracking
- Returns and refunds
- Product information
- Loyalty rewards and discounts
- General customer support questions

Use the appropriate tools whenever they are needed.

Use the knowledge base for:
- Product specifications
- Return policies
- Warranty information
- Loyalty program information
- Order status definitions

Use the Gateway tools for:
- Order lookup
- Customer lookup
- Refund processing

Use the loyalty discount tool for loyalty calculations.

Use the browser when the customer asks you to browse a website
or retrieve information from the web.

Use customer context from memory when it is relevant.

Do not invent order information, refund information, product details,
or loyalty calculations.
""",
            )

            # 7. Invoke the agent
            response = agent(user_input)

        # 8. Return the response text
        if response and response.message:
            content = response.message.get("content", [])

            if content:
                first_block = content[0]

                if isinstance(first_block, dict):
                    return first_block.get("text", str(first_block))

                return str(first_block)

        return str(response)

    except Exception as e:
        logger.exception("Agent invocation failed")
        return f"Sorry, I encountered an error while processing your request: {e}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()
