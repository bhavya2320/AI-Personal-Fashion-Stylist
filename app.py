
import os
import re
import streamlit as st

from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Personal Fashion Stylist",
    page_icon="👗",
    layout="centered"
)


# ============================================================
# API KEY
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API key is not configured.")
    st.stop()


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# STATE
# ============================================================

class FashionState(TypedDict):
    user_request: str
    budget: float

    fashion_analysis: str
    estimated_cost: float
    budget_valid: bool

    optimized_outfit: str
    final_recommendation: str


# ============================================================
# CONTENT HANDLER
# ============================================================

def extract_content(response):

    if isinstance(response.content, list):

        parts = []

        for block in response.content:

            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))

        return "\n".join(parts)

    return str(response.content)


# ============================================================
# NODE 1 — FASHION ANALYZER
# ============================================================

def fashion_analyzer(state: FashionState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert AI Personal Fashion Stylist.

Analyze the user's request and create ONE complete fashion
recommendation.

Include:

- Occasion
- Preferred style
- Preferred colors
- Clothing type
- Main outfit
- Footwear
- Accessories
- Estimated price of each item
- TOTAL COST

The user's maximum budget is ₹{budget}.

Try to keep the total cost within the budget.

At the end write exactly:

TOTAL COST: ₹<number>
"""
        ),
        (
            "human",
            "{user_request}"
        )
    ])

    response = (prompt | llm).invoke({
        "user_request": state["user_request"],
        "budget": state["budget"]
    })

    result = extract_content(response)

    state["fashion_analysis"] = result

    matches = re.findall(
        r"TOTAL COST:\s*₹?\s*([\d,]+(?:\.\d+)?)",
        result,
        re.IGNORECASE
    )

    if matches:
        state["estimated_cost"] = float(
            matches[-1].replace(",", "")
        )
    else:
        state["estimated_cost"] = 0.0

    state["budget_valid"] = (
        state["estimated_cost"] > 0
        and state["estimated_cost"] <= state["budget"]
    )

    return state


# ============================================================
# NODE 2 — BUDGET OPTIMIZER
# ============================================================

def budget_optimizer(state: FashionState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an affordable fashion stylist.

The previous outfit exceeded the user's budget.

Create a cheaper alternative while preserving:

- Occasion
- Preferred style
- Preferred colors
- Clothing preference

Replace expensive items with affordable alternatives.

The new outfit should stay within the user's budget.

At the end write exactly:

TOTAL COST: ₹<number>
"""
        ),
        (
            "human",
            """
User Request:
{user_request}

Maximum Budget:
₹{budget}

Previous Recommendation:
{fashion_analysis}
"""
        )
    ])

    response = (prompt | llm).invoke({
        "user_request": state["user_request"],
        "budget": state["budget"],
        "fashion_analysis": state["fashion_analysis"]
    })

    state["optimized_outfit"] = extract_content(response)

    return state


# ============================================================
# NODE 3 — FINAL RESULT
# ============================================================

def final_result(state: FashionState):

    if state["optimized_outfit"]:
        state["final_recommendation"] = (
            state["optimized_outfit"]
        )
    else:
        state["final_recommendation"] = (
            state["fashion_analysis"]
        )

    return state


# ============================================================
# LANGGRAPH
# ============================================================

workflow = StateGraph(FashionState)

workflow.add_node(
    "fashion_analyzer",
    fashion_analyzer
)

workflow.add_node(
    "budget_optimizer",
    budget_optimizer
)

workflow.add_node(
    "final_result",
    final_result
)


workflow.add_edge(
    START,
    "fashion_analyzer"
)


def budget_decision(state: FashionState):

    if state["budget_valid"]:
        return "within_budget"

    return "over_budget"


workflow.add_conditional_edges(
    "fashion_analyzer",
    budget_decision,
    {
        "within_budget": "final_result",
        "over_budget": "budget_optimizer"
    }
)


workflow.add_edge(
    "budget_optimizer",
    "final_result"
)

workflow.add_edge(
    "final_result",
    END
)


fashion_agent = workflow.compile()


# ============================================================
# RUN AGENT
# ============================================================

def run_fashion_agent(user_request, budget):

    initial_state: FashionState = {

        "user_request": user_request,

        "budget": float(budget),

        "fashion_analysis": "",

        "estimated_cost": 0.0,

        "budget_valid": False,

        "optimized_outfit": "",

        "final_recommendation": ""
    }

    result = fashion_agent.invoke(initial_state)

    return result["final_recommendation"]


# ============================================================
# USER INTERFACE
# ============================================================

st.title("👗 AI Personal Fashion Stylist")

st.write(
    "Get a personalized outfit recommendation based on "
    "your occasion, style, preferences and budget."
)

st.divider()


occasion = st.selectbox(
    "🎉 Occasion",
    [
        "Birthday Party",
        "Wedding",
        "Family Function",
        "College Event",
        "Office",
        "Date",
        "Casual Outing",
        "Festival",
        "Other"
    ]
)


style = st.selectbox(
    "✨ Preferred Style",
    [
        "Elegant",
        "Casual",
        "Trendy",
        "Traditional",
        "Minimal",
        "Party Wear",
        "Formal"
    ]
)


color = st.text_input(
    "🎨 Preferred Color",
    placeholder="Example: Black, Pink, Navy Blue"
)


clothing = st.text_input(
    "👗 Clothing Preference",
    placeholder="Example: Dress, Saree, Kurti, Jeans"
)


budget = st.number_input(
    "💰 Maximum Budget (₹)",
    min_value=500,
    max_value=100000,
    value=3000,
    step=500
)


requirements = st.text_area(
    "📝 Additional Requirements",
    placeholder=(
        "Example: Comfortable, elegant, not too flashy..."
    )
)


if st.button(
    "✨ Create My Outfit",
    use_container_width=True
):

    user_request = f"""
Create a fashion recommendation for me.

Occasion: {occasion}

Preferred Style: {style}

Preferred Color: {color}

Clothing Preference: {clothing}

Maximum Budget: ₹{budget}

Additional Requirements:
{requirements}
"""

    with st.spinner(
        "👗 Your personal stylist is creating your look..."
    ):

        try:

            recommendation = run_fashion_agent(
                user_request,
                budget
            )

            st.success(
                "Your personalized outfit is ready! ✨"
            )

            st.markdown(recommendation)

        except Exception as e:

            st.error(
                "Unable to generate the recommendation. "
                "Please try again later."
            )

            st.code(str(e))


st.divider()

st.caption(
    "Powered by Gemini • LangChain • LangGraph"
)
