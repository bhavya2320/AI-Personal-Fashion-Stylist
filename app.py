
import os
import streamlit as st
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
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
# CUSTOM UI
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #ffffff;
}

h1 {
    text-align: center;
    color: #111111;
}

.subtitle {
    text-align: center;
    color: #666666;
    font-size: 17px;
    margin-bottom: 30px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 17px;
    font-weight: bold;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #f7f7f7;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.title("👗 AI Personal Fashion Stylist")

st.markdown(
    '<p class="subtitle">Get a personalized outfit recommendation based on your style, occasion and budget.</p>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT UI
# ============================================================

col1, col2 = st.columns(2)

with col1:
    occasion = st.selectbox(
        "🎉 Occasion",
        [
            "College",
            "Casual Outing",
            "Party",
            "Wedding",
            "Interview",
            "Date",
            "Festival"
        ]
    )

    style = st.selectbox(
        "✨ Preferred Style",
        [
            "Casual",
            "Trendy",
            "Elegant",
            "Traditional",
            "Formal",
            "Streetwear"
        ]
    )

    color = st.selectbox(
        "🎨 Preferred Color",
        [
            "Black",
            "White",
            "Blue",
            "Pink",
            "Red",
            "Green",
            "Any Color"
        ]
    )

with col2:
    clothing = st.selectbox(
        "👚 Clothing Preference",
        [
            "Western",
            "Indian",
            "Indo-Western",
            "Any"
        ]
    )

    budget = st.number_input(
        "💰 Maximum Budget (₹)",
        min_value=500,
        max_value=100000,
        value=2000,
        step=500
    )

    requirements = st.text_area(
        "📝 Additional Requirements",
        placeholder="Example: Comfortable, trendy, suitable for summer..."
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
# GEMINI
# ============================================================

api_key = os.environ.get("GOOGLE_API_KEY")

if api_key:

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key
    )

else:
    llm = None


# ============================================================
# HELPER
# ============================================================

def extract_content(response):

    if hasattr(response, "content"):
        return response.content

    return str(response)


# ============================================================
# GEMINI FASHION ANALYZER
# ============================================================

def fashion_analyzer(state: FashionState):

    prompt = f"""
You are an expert personal fashion stylist.

Create a practical outfit recommendation using:

Occasion: {occasion}
Preferred Style: {style}
Preferred Color: {color}
Clothing Preference: {clothing}
Maximum Budget: ₹{budget}
Additional Requirements: {requirements}

Give:

1. Outfit
2. Top/Shirt/Kurti suggestion
3. Bottom suggestion
4. Footwear
5. Accessories
6. Estimated total cost
7. Styling tip

Keep the recommendation practical and within the budget.

Return a clean, easy-to-read answer.
"""

    # Try Gemini
    if llm:

        try:

            response = llm.invoke(prompt)

            content = extract_content(response)

            return {
                **state,
                "fashion_analysis": content,
                "estimated_cost": budget * 0.8,
                "budget_valid": True
            }

        except Exception as e:

            # Gemini quota / API error
            error_text = str(e)

            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:

                fallback = f"""
### 👗 Recommended Outfit

**Occasion:** {occasion}  
**Style:** {style}  
**Color:** {color}

**👚 Outfit:**  
Choose a stylish {clothing.lower()} outfit suitable for {occasion.lower()}.

**👖 Bottom:**  
Pair it with a comfortable and well-fitted bottom that complements the outfit.

**👟 Footwear:**  
Choose clean sneakers, flats or simple sandals depending on the outfit.

**👜 Accessories:**  
Add minimal accessories such as a watch, bracelet, earrings or a simple handbag.

**💰 Estimated Budget:** ₹{int(budget * 0.75)}–₹{int(budget)}

**✨ Styling Tip:**  
Keep the overall look balanced. Since you prefer a {style.lower()} style, avoid over-accessorizing.

> ℹ️ Gemini is temporarily unavailable because the API free-tier quota has been reached. This recommendation was generated using the app's built-in fallback system.
"""

                return {
                    **state,
                    "fashion_analysis": fallback,
                    "estimated_cost": budget * 0.75,
                    "budget_valid": True
                }

            else:

                fallback = f"""
### 👗 Outfit Recommendation

For your **{occasion}** occasion, try a **{style} {clothing} outfit** in **{color}**.

Keep the outfit comfortable and within your **₹{int(budget)}** budget.

**Accessories:** Minimal accessories  
**Footwear:** Comfortable footwear matching the outfit  
**Tip:** Choose well-fitted clothing and keep the colors balanced.
"""

                return {
                    **state,
                    "fashion_analysis": fallback,
                    "estimated_cost": budget * 0.75,
                    "budget_valid": True
                }

    # No API key
    fallback = f"""
### 👗 Outfit Recommendation

**Occasion:** {occasion}

**Style:** {style}

**Color:** {color}

**Clothing:** {clothing}

**Budget:** ₹{int(budget)}

Choose a comfortable {style.lower()} outfit suitable for {occasion.lower()}.
Pair it with simple accessories and matching footwear.

**Styling Tip:** Keep the outfit balanced and comfortable.
"""

    return {
        **state,
        "fashion_analysis": fallback,
        "estimated_cost": budget * 0.75,
        "budget_valid": True
    }


# ============================================================
# BUDGET OPTIMIZER
# ============================================================

def budget_optimizer(state: FashionState):

    if state["estimated_cost"] <= state["budget"]:
        return state

    if llm:

        try:

            prompt = f"""
Optimize this outfit so that it stays within ₹{state['budget']}.

Current recommendation:

{state['fashion_analysis']}

Give a more affordable version while maintaining the same style.
"""

            response = llm.invoke(prompt)

            return {
                **state,
                "optimized_outfit": extract_content(response)
            }

        except Exception:
            pass

    return {
        **state,
        "optimized_outfit": state["fashion_analysis"]
    }


# ============================================================
# FINAL RESULT
# ============================================================

def final_result(state: FashionState):

    recommendation = state.get("optimized_outfit")

    if not recommendation:
        recommendation = state["fashion_analysis"]

    return {
        **state,
        "final_recommendation": recommendation
    }


# ============================================================
# LANGGRAPH
# ============================================================

workflow = StateGraph(FashionState)

workflow.add_node("fashion_analyzer", fashion_analyzer)
workflow.add_node("budget_optimizer", budget_optimizer)
workflow.add_node("final_result", final_result)

workflow.add_edge(START, "fashion_analyzer")
workflow.add_edge("fashion_analyzer", "budget_optimizer")
workflow.add_edge("budget_optimizer", "final_result")
workflow.add_edge("final_result", END)

fashion_agent = workflow.compile()


# ============================================================
# BUTTON
# ============================================================

st.markdown("---")

if st.button("✨ Create My Outfit"):

    user_request = f"""
    Occasion: {occasion}
    Style: {style}
    Color: {color}
    Clothing: {clothing}
    Budget: ₹{budget}
    Requirements: {requirements}
    """

    initial_state = {
        "user_request": user_request,
        "budget": float(budget),
        "fashion_analysis": "",
        "estimated_cost": 0.0,
        "budget_valid": True,
        "optimized_outfit": "",
        "final_recommendation": ""
    }

    with st.spinner("✨ Creating your personalized outfit..."):

        result = fashion_agent.invoke(initial_state)

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )

    st.markdown(result["final_recommendation"])

    st.markdown("</div>", unsafe_allow_html=True)

    st.success("✨ Outfit recommendation generated successfully!")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Powered by Gemini • LangChain • LangGraph • Streamlit"
)
