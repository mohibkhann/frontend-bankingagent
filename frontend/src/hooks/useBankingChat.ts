// src/hooks/useBankingChat.ts
import { useState, useRef, useCallback, useEffect } from 'react';
import APIClient, { ChatResponse } from '../lib/api-client';

export interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  agent_used?: string;
  session_id?: string;
}
export interface UseBankingChatReturn {
  messages: Message[];
  isLoading: boolean;
  isConnected: boolean;
  error: string | null;
  sendMessage: (message: string) => Promise<void>;
  clearMessages: () => void;
  reconnect: () => void;
  connectionType: 'http' | 'websocket';
  setConnectionType: (type: 'http' | 'websocket') => void;
}

export const useBankingChat = (clientId: number = 430): UseBankingChatReturn => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);6
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionType, setConnectionType] = useState<'http' | 'websocket'>('http');
  const [sessionId, setSessionId] = useState<string>('');

  const apiClient = useRef(new APIClient('http://localhost:8000', clientId));
  const websocket = useRef<WebSocket | null>(null);

  // Generate session ID on mount
  useEffect(() => {
    const newSessionId = `session_${clientId}_${Date.now()}`;
    setSessionId(newSessionId);
  }, [clientId]);

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await apiClient.current.checkHealth();
        setIsConnected(health.status === 'online');
        setError(null);
      } catch (err) {
        console.log(err)
        setError('Failed to connect to API');
        setIsConnected(false);
      }
    };

    checkHealth();
  }, []);

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    if (websocket.current) {
      websocket.current.close();
    }

    websocket.current = apiClient.current.createWebSocketConnection(
      (data: ChatResponse) => {
        const assistantMessage: Message = {
            id: `msg_${Date.now()}_assistant`,
            type: 'assistant',  // Make sure this is set
            content: data.response,  
            timestamp: new Date(),
            agent_used: data.agent_used,
            session_id: data.session_id
          };

        setMessages(prev => [...prev, assistantMessage]);
        setIsLoading(false);
      },
      (error) => {
        console.error('WebSocket error:', error);
        setError('WebSocket connection error');
        setIsConnected(false);
        setIsLoading(false);
      },
      () => {
        setIsConnected(false);
        setIsLoading(false);
      }
    );

    // Set connected when WebSocket opens
    websocket.current.onopen = () => {
      setIsConnected(true);
      setError(null);
    };
  }, []);

  // Initialize WebSocket connection when connection type changes
  useEffect(() => {
    if (connectionType === 'websocket') {
      connectWebSocket();
    } else if (websocket.current) {
      websocket.current.close();
      websocket.current = null;
      setIsConnected(true); // For HTTP mode
    }

    return () => {
      if (websocket.current) {
        websocket.current.close();
        websocket.current = null;
      }
    };
  }, [connectionType, connectWebSocket]);

  const sendMessage = useCallback(async (message: string) => {
    if (!message.trim()) return;

    const userMessage: Message = {
      id: `msg_${Date.now()}_user`,
      type: 'user',
      content: message.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      if (connectionType === 'websocket' && websocket.current) {
        // WebSocket mode
        apiClient.current.sendWebSocketMessage(websocket.current, message);
      } else {
        // HTTP mode
        const response = await apiClient.current.sendMessage(message, sessionId);
        console.log("response: ", response)
        const assistantMessage: Message = {
          id: `msg_${Date.now()}_assistant`,
          type: 'assistant',
          content: response.response,
          timestamp: new Date(),
          agent_used: response.agent_used,
          session_id: response.session_id
        };

        setMessages(prev => [...prev, assistantMessage]);
        setIsLoading(false);

        // Update session ID if changed
        if (response.session_id !== sessionId) {
          setSessionId(response.session_id);
        }
      }
    } catch (err) {
      console.error('Failed to send message:', err);
      setError(err instanceof Error ? err.message : 'Failed to send message');
      setIsLoading(false);
      
      // Add error message to chat
      const errorMessage: Message = {
        id: `msg_${Date.now()}_error`,
        type: 'assistant',
        content: '❌ Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  }, [connectionType, sessionId]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
    // Generate new session ID
    const newSessionId = `session_${clientId}_${Date.now()}`;
    setSessionId(newSessionId);
  }, [clientId]);

  const reconnect = useCallback(async () => {
    setError(null);
    setIsConnected(false);

    if (connectionType === 'websocket') {
      connectWebSocket();
    } else {
      try {
        const health = await apiClient.current.checkHealth();
        setIsConnected(health.status === 'online');
      } catch (err) {
        setError('Failed to reconnect to API');
      }
    }
  }, [connectionType, connectWebSocket]);

  return {
    messages,
    isLoading,
    isConnected,
    error,
    sendMessage,
    clearMessages,
    reconnect,
    connectionType,
    setConnectionType,
  };
};