

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
from strands.agent.conversation_manager import SummarizingConversationManager
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CSAI_Agent")


app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"



GATEWAY_URL = "https://customersupportgateway-630q2cntc0.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "CDAKLIHF8V"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-22B3IX24wm"

class DiscountBreakdown(BaseModel):
    points_redeemed: int = Field(ge=0)
    points_value: float = Field(ge=0)
    tier_discount_pct: float = Field(ge=0)
    tier_discount: float = Field(ge=0)
    final_total: float = Field(ge=0)
    total_savings: float = Field(ge=0)
    points_earned: int = Field(ge=0)
    remaining_points: int = Field(ge=0)
    fallback: bool = False





model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)

memory_client = MemoryClient(region_name=REGION)

_bedrock_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
)



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



class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
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



@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter.

    If the Code Interpreter is unavailable, the tool falls back to
    a tier-only discount calculation without redeeming points.

    Args:
        loyalty_points: Customer's current points balance
        tier: Customer tier — Silver, Gold, or Platinum
        order_total: Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Structured discount breakdown and final price.
    """

    # Business rules executed inside the sandboxed Code Interpreter.
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

# Redeem points in multiples of 500.
# 100 points = $1, so the redemption cannot exceed 50% of the order value.
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
    "remaining_points": remaining_points,
    "fallback": False
}}

print(json.dumps(result))
"""

    try:
        
        with code_session(REGION) as session:
        # with code_session('invalid-region') as session:
            response = session.invoke(
                "executeCode",
                {
                    "language": "python",
                    "code": code,
                    "clearContext": True,
                },
            )

        if not response:
            raise RuntimeError(
                "Code Interpreter returned no result"
            )

        first_result = (
            response[0]
            if isinstance(response, list)
            else response
        )

        if isinstance(first_result, str):
            raw_result = first_result
        else:
            raw_result = json.dumps(first_result)

        parsed_result = json.loads(raw_result)

        validated_result = DiscountBreakdown.model_validate(
            parsed_result
        )

        return validated_result.model_dump_json()

    except Exception as exc:
        # Fallback path:
        # Code Interpreter is unavailable or failed.
        # No loyalty points are redeemed here.
        # Only the customer's tier discount is applied.
        logger.warning(
            "Code Interpreter unavailable; using tier-only fallback: %s",
            exc,
        )

        tier_rates = {
            "Silver": 0.00,
            "Gold": 0.10,
            "Platinum": 0.15,
        }

        tier_rate = tier_rates.get(tier, 0.00)

        points_redeemed = 0
        points_value = 0.0

        tier_discount = order_total * tier_rate

        final_total = max(
            order_total - tier_discount,
            0,
        )

        total_savings = tier_discount

        points_earned = 0
        remaining_points = loyalty_points

        fallback_result = DiscountBreakdown(
            points_redeemed=points_redeemed,
            points_value=points_value,
            tier_discount=round(tier_discount, 2),
            final_total=round(final_total, 2),
            total_savings=round(total_savings, 2),
            points_earned=points_earned,
            remaining_points=remaining_points,
            fallback=True,
        )

        return fallback_result.model_dump_json()
 


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
        conversation_history = payload.get("messages", [])

        if not user_input:
            return "Please provide a prompt."

        # 2. Create the memory hook
        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        conversation_manager = SummarizingConversationManager(
            summary_ratio=0.3,
            preserve_recent_messages=10,
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

        try:
            with gateway_client:
                try:
                    gateway_tools = gateway_client.list_tools_sync()
                    tools.extend(gateway_tools)

                    logger.info(
                        "Gateway connected successfully. Loaded %d tools.",
                        len(gateway_tools),
                    )

                except TimeoutError:
                    logger.exception("Gateway tool loading timed out")

                except ConnectionError:
                    logger.exception("Gateway connection failed")

                except Exception as exc:
                    logger.exception(
                        "Gateway tool loading failed: %s", exc
                    )

        except Exception as exc:
            logger.exception(
                "Gateway connection failed during setup: %s",
                exc,
            )

        # 6. Create the Strands Agent
        agent = Agent(
            model=model,
            tools=tools,
            hooks=[memory_hook],
            messages=conversation_history,
            conversation_manager=conversation_manager,
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
