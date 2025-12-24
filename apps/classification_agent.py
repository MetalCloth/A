from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import google.generativeai as genai
import json
import operator

class VideoAnalysisState(TypedDict):
    """State for the video analysis agent"""
    source_metadata: Dict
    analysis_results: List[Dict]
    classifications: List[Dict]
    current_index: int
    error: str

class ClassificationAgent:
    """LangGraph agent for video classification using Gemini"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Also setup LangChain version for potential future use
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=1500
        )
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(VideoAnalysisState)
        
        # Add nodes
        workflow.add_node("prepare_data", self._prepare_data)
        workflow.add_node("classify_video", self._classify_single_video)
        workflow.add_node("aggregate_results", self._aggregate_results)
        
        # Add edges
        workflow.set_entry_point("prepare_data")
        workflow.add_edge("prepare_data", "classify_video")
        workflow.add_conditional_edges(
            "classify_video",
            self._should_continue,
            {
                "continue": "classify_video",
                "end": "aggregate_results"
            }
        )
        workflow.add_edge("aggregate_results", END)
        
        return workflow.compile()
    
    def _prepare_data(self, state: VideoAnalysisState) -> VideoAnalysisState:
        """Prepare data for classification"""
        state["classifications"] = []
        state["current_index"] = 0
        return state
    
    def _classify_single_video(self, state: VideoAnalysisState) -> VideoAnalysisState:
        """Classify a single video match using Gemini"""
        idx = state["current_index"]
        
        if idx >= len(state["analysis_results"]):
            return state
        
        result = state["analysis_results"][idx]
        
        # Prepare prompt for Gemini
        prompt = f"""You are an expert video content analyst. Classify the relationship between two videos based on similarity metrics.

Source Video: "{state['source_metadata']['title']}"
Channel: {state['source_metadata']['channel']}
Duration: {state['source_metadata']['duration']}s

Candidate Video: "{result['title']}"
Channel: {result['channel']}
Duration: {state['source_metadata']['duration'] - result['duration_diff']}s

Metrics:
- Visual Similarity: {result['visual_similarity']*100:.1f}%
- Title Similarity: {result['title_similarity']*100:.1f}%
- Duration Difference: {result['duration_diff']}s
- Same Channel: {result['same_channel']}

Classification Rules:
- **Re-upload**: Exact same content (>95% visual, <2s duration diff)
- **Edited Copy**: Same base content with modifications (85-95% visual)
- **Unrelated**: Different content (<85% visual)

Provide ONLY a JSON response (no markdown, no explanation):
{{
    "classification": "Re-upload/Edited Copy/Unrelated",
    "confidence": "High/Medium/Low",
    "reasoning": "Brief explanation in 1-2 sentences"
}}"""
        
        try:
            # Call Gemini API
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Clean up response (remove markdown if present)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            classification = json.loads(response_text)
            state["classifications"].append(classification)
                
        except Exception as e:
            # Fallback to rule-based
            print(f"Gemini API error: {e}, using fallback")
            state["classifications"].append(self._fallback_classification(result))
        
        state["current_index"] += 1
        return state
    
    def _fallback_classification(self, result: Dict) -> Dict:
        """Rule-based fallback classification"""
        visual = result['visual_similarity']
        duration = result['duration_diff']
        
        if visual > 0.95 and duration < 2:
            return {
                "classification": "Re-upload",
                "confidence": "High",
                "reasoning": "Nearly identical visual content and duration"
            }
        elif visual > 0.85:
            return {
                "classification": "Edited Copy",
                "confidence": "Medium",
                "reasoning": "High visual similarity with some differences"
            }
        else:
            return {
                "classification": "Unrelated",
                "confidence": "Low",
                "reasoning": "Visual similarity may be coincidental"
            }
    
    def _should_continue(self, state: VideoAnalysisState) -> str:
        """Decide whether to continue classifying"""
        if state["current_index"] < len(state["analysis_results"]):
            return "continue"
        return "end"
    
    def _aggregate_results(self, state: VideoAnalysisState) -> VideoAnalysisState:
        """Merge classifications with analysis results"""
        for i, result in enumerate(state["analysis_results"]):
            if i < len(state["classifications"]):
                result.update(state["classifications"][i])
        return state
    
    def classify(self, source_metadata: Dict, analysis_results: List[Dict]) -> List[Dict]:
        """Run the classification workflow"""
        initial_state = {
            "source_metadata": source_metadata,
            "analysis_results": analysis_results,
            "classifications": [],
            "current_index": 0,
            "error": ""
        }
        
        final_state = self.graph.invoke(initial_state)
        return final_state["analysis_results"]
