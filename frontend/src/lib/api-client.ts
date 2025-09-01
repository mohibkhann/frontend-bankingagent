// src/lib/api-client.ts
export interface ChatMessage {
  message: string;
  client_id?: number;
  session_id?: string;
}

export interface ChatResponse {
  message: string;
  response: string;
  agent_used: string;
  timestamp: string;
  session_id: string;
  success: boolean;
  error?: string;
  follow_up_info?: {
    was_follow_up: boolean;
    required_agent_execution: boolean;
    target_agent?: string;
    response_strategy?: string;
    enhanced_query?: string;
  };
}

export interface HealthResponse {
  status: string;
  router_status: string;
  timestamp: string;
}

class APIClient {
  private baseURL: string;
  private clientId: number;

  constructor(baseURL: string = 'http://localhost:8000', clientId: number = 1) {
    this.baseURL = baseURL;
    this.clientId = clientId;
  }

  // Health check
  async checkHealth(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseURL}/api/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }
    return await response.json();
  }

  // Send chat message
  async sendMessage(
    message: string, 
    sessionId?: string
  ): Promise<ChatResponse> {
    const payload: ChatMessage = {
      message,
      client_id: this.clientId,
      session_id: sessionId
    };

    const response = await fetch(`${this.baseURL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Request failed: ${response.statusText}`);
    }

    return await response.json();
  }

  // Get agents status
  async getAgentsStatus() {
    const response = await fetch(`${this.baseURL}/api/agents/status`);
    if (!response.ok) {
      throw new Error(`Failed to get agents status: ${response.statusText}`);
    }
    return await response.json();
  }

  // WebSocket connection
  createWebSocketConnection(
    onMessage: (data: ChatResponse) => void,
    onError?: (error: Event) => void,
    onClose?: () => void
  ): WebSocket {
    const wsURL = `ws://localhost:8000/api/ws/${this.clientId}`;
    const ws = new WebSocket(wsURL);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) onError(error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      if (onClose) onClose();
    };

    return ws;
  }

  // Send message via WebSocket
  sendWebSocketMessage(ws: WebSocket, message: string) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ message }));
    } else {
      throw new Error('WebSocket is not connected');
    }
  }
}

export default APIClient;