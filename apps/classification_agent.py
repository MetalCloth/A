from typing import Dict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
import os
from dotenv import load_dotenv
from config.settings import settings

load_dotenv()

os.environ['GEMINI_API_KEY']=os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1,
    max_output_tokens=1500
)

# --- Graph Functions ---
def prepare_data(state):
    state["classifications"] = []
    state["current_index"] = 0
    return state

def classify_single_video_node(state):
    idx = state["current_index"]
    
    if idx >= len(state["analysis_results"]):
        return state
    
    result = state["analysis_results"][idx]
    source_info = state["source_metadata"]
    
    # Construct the prompt string
    prompt_text = f"""You are an expert video content analyst. Classify the relationship between two videos based on similarity metrics.

Source Video: "{source_info['title']}"
Channel: {source_info['channel']}
Duration: {source_info['duration']}s

Candidate Video: "{result['title']}"
Channel: {result['channel']}
Duration: {source_info['duration'] - result['duration_diff']}s

Metrics:
- Visual Similarity: {result['visual_match']*100:.1f}%
- Title Similarity: {result['title_match']*100:.1f}%
- Duration Difference: {result['duration_diff']}s
- Same Channel: {result['same_channel']}

Classification Rules:
- **Re-upload**: Exact same content (>95% visual, <2s duration diff)
- **Edited Copy**: Same base content with modifications (85-95% visual)
- **Unrelated**: Different content (<85% visual)

Provide ONLY a JSON response (no markdown, no explanation):
{{
    "label": "Re-upload/Edited Copy/Unrelated",
    "confidence": "High/Medium/Low",
    "reason": "Brief explanation in 1-2 sentences"
}}"""
    
    try:
        response = llm.invoke(prompt_text)
        
        response_text = response.content.strip()
        
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        response_text = response_text.strip()
        
        # Parse JSON
        classification = json.loads(response_text)
        state["classifications"].append(classification)
            
    except Exception as e:
        print(f"LLM Invocation/Parsing error: {e}, using fallback")
        state["classifications"].append(fallback_classification(result))
    
    state["current_index"] += 1
    return state

def fallback_classification(result):
    visual = result['visual_match']
    duration = result['duration_diff']
    
    if visual > 0.95 and duration < 2:
        return {
            "label": "Re-upload",
            "confidence": "High",
            "reason": "Nearly identical visual content and duration"
        }
    elif visual > 0.85:
        return {
            "label": "Edited Copy",
            "confidence": "Medium",
            "reason": "High visual similarity with some differences"
        }
    else:
        return {
            "label": "Unrelated",
            "confidence": "Low",
            "reason": "Visual similarity may be coincidental"
        }

def should_continue(state):
    if state["current_index"] < len(state["analysis_results"]):
        return "continue"
    return "end"

def aggregate_results(state):
    for i, result in enumerate(state["analysis_results"]):
        if i < len(state["classifications"]):
            result.update(state["classifications"][i])
    return state

# --- Graph Construction ---
def build_graph():
    workflow = StateGraph(dict)
    
    workflow.add_node("prepare_data", prepare_data)
    workflow.add_node("classify_video", classify_single_video_node)
    workflow.add_node("aggregate_results", aggregate_results)
    
    workflow.set_entry_point("prepare_data")
    workflow.add_edge("prepare_data", "classify_video")
    workflow.add_conditional_edges(
        "classify_video",
        should_continue,
        {
            "continue": "classify_video",
            "end": "aggregate_results"
        }
    )
    workflow.add_edge("aggregate_results", END)
    
    return workflow.compile()

graph = build_graph()

def classify_with_ai(source_metadata, analysis_results):
    initial_state = {
        "source_metadata": source_metadata,
        "analysis_results": analysis_results,
        "classifications": [],
        "current_index": 0,
        "error": ""
    }
    
    final_state = graph.invoke(initial_state)
    return final_state["analysis_results"]