from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Import your existing agents
from agents.agent_router import EnhancedPersonalFinanceRouter

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GX Banking AI API",
    description="AI-powered banking assistant API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str
    client_id: Optional[int] = 430 
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    response: str
    agent_used: str
    timestamp: str
    session_id: str
    success: bool
    error: Optional[str] = None
    follow_up_info: Optional[Dict[str, Any]] = None

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str

# Global router instance
router_instance: Optional[EnhancedPersonalFinanceRouter] = None

# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the router on startup"""
    global router_instance
    try:
        # Update these paths to match your data files
        client_csv_path = "data/client_data.csv"  # Update path
        overall_csv_path = "data/overall_data.csv"  # Update path
        
        logger.info("Initializing Enhanced Personal Finance Router...")
        router_instance = EnhancedPersonalFinanceRouter(
            client_csv_path=client_csv_path,
            overall_csv_path=overall_csv_path,
            model_name="gpt-4o"
        )
        logger.info("✅ Router initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize router: {e}")
        router_instance = None

@app.get("/", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy" if router_instance else "unhealthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/api/health")
async def api_health():
    """API health endpoint for frontend"""
    return {
        "status": "online",
        "router_status": "initialized" if router_instance else "not_initialized",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message_data: ChatMessage):
    """Main chat endpoint"""
    if not router_instance:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    try:
        logger.info(f"Processing message: {message_data.message}")
        
        # Process the message using your existing router
        result = router_instance.chat(
            client_id=message_data.client_id,
            user_query=message_data.message,
            session_id=message_data.session_id
        )
        
        return ChatResponse(
            message=message_data.message,
            response=result["response"],
            agent_used=result.get("agent_used", "unknown"),
            timestamp=result.get("timestamp", datetime.now().isoformat()),
            session_id=result.get("session_id", ""),
            success=result.get("success", True),
            error=result.get("error"),
            follow_up_info=result.get("follow_up_info")
        )
        
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.websocket("/api/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket)
    session_id = f"ws_{client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_json = json.loads(data)
            
            if not router_instance:
                await manager.send_personal_message({
                    "error": "AI service not available"
                }, websocket)
                continue
            
            try:
                # Process message
                result = router_instance.chat(
                    client_id=client_id,
                    user_query=message_json["message"],
                    session_id=session_id
                )
                
                # Send response back
                response = {
                    "message": message_json["message"],
                    "response": result["response"],
                    "agent_used": result.get("agent_used", "unknown"),
                    "timestamp": result.get("timestamp", datetime.now().isoformat()),
                    "session_id": result.get("session_id", session_id),
                    "success": result.get("success", True)
                }
                
                await manager.send_personal_message(response, websocket)
                
            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                await manager.send_personal_message({
                    "error": f"Processing error: {str(e)}"
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"Client {client_id} disconnected")

@app.get("/api/agents/status")
async def get_agents_status():
    """Get status of all agents"""
    if not router_instance:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    return {
        "router_status": "active",
        "agents": {
            "spending_agent": "active",
            "budget_agent": "active", 
            "rag_agent": "active"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)