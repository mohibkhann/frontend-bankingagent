// src/components/ui/Chatmessage?.tsx
import { Message } from '@/context/BankingChatContext';
import React, { useEffect } from 'react';

interface ChatMessageProps {
  message: Message;
  isLoading?: boolean;
}



const ChatMessage: React.FC<ChatMessageProps> = ({ message, isLoading = false }) => {
  const formatResponse = (text: string) => {
  return text
    .replace(/\\n\\n/g, '\n\n')  // Fix double newlines
    .replace(/\\n/g, '\n')       // Fix single newlines
    .replace(/\*\*(.*?)\*\*/g, '$1')  // Remove **bold** markers
    .replace(/^\d+\.\s/gm, '• ')      // Replace numbered lists with bullets
    .trim();
};
  const isUser = message?.type === 'user';
  console.log("message: ", message)
  useEffect(()=> {
    console.log("message: ", message)
  }, [message]
)
  // const isUser = true
  // Format timestamp
  const timeString = message?.timestamp.toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  });

  // Get agent badge color
  const getAgentBadgeColor = (agent?: string) => {
    switch (agent) {
      case 'spending': return 'bg-blue-100 text-blue-800';
      case 'budget': return 'bg-green-100 text-green-800';
      case 'rag': return 'bg-purple-100 text-purple-800';
      case 'follow_up': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  // Format agent name for display
  const formatAgentName = (agent?: string) => {
    switch (agent) {
      case 'spending': return 'Spending Analysis';
      case 'budget': return 'Budget Planning';
      case 'rag': return 'Banking Services';
      case 'follow_up': return 'Follow-up';
      default: return 'AI Assistant';
    }
  };

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-3 relative ${
          isUser
            ? 'bg-blue-600 text-white'
            : 'bg-white text-gray-800 border border-gray-200 shadow-sm'
        }`}
      >
        {/* Message Content */}
        <div className="break-words">
          {isLoading ? (
            <div className="flex items-center space-x-2">
              <div className="animate-spin w-4 h-4 border-2 border-gray-300 border-t-blue-600 rounded-full"></div>
              <span className="text-sm text-gray-500">AI is thinking...</span>
            </div>
          ) : (
              <div className="whitespace-pre-wrap">{formatResponse(message?.content || '')}</div>

          )}
        </div>

        {/* Message metadata */}
        <div className={`flex items-center justify-between mt-2 text-xs ${
          isUser ? 'text-blue-100' : 'text-gray-500'
        }`}>
          <div className="flex items-center space-x-2">
            {/* Agent badge for assistant messages */}
            {!isUser && message?.agent_used && (
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                getAgentBadgeColor(message?.agent_used)
              }`}>
                {formatAgentName(message?.agent_used)}
              </span>
            )}
            
            {/* Session ID (for debugging, can be removed in production) */}
            {message?.session_id && (
              <span className="opacity-50 font-mono text-xs">
                Session: {message?.session_id.slice(-6)}
              </span>
            )}
          </div>

          {/* Timestamp */}
          <span className="opacity-75">{timeString}</span>
        </div>

        {/* Triangle pointer */}
        <div
          className={`absolute top-3 w-0 h-0 ${
            isUser
              ? 'right-[-6px] border-l-[6px] border-l-blue-600 border-t-[6px] border-t-transparent border-b-[6px] border-b-transparent'
              : 'left-[-6px] border-r-[6px] border-r-white border-t-[6px] border-t-transparent border-b-[6px] border-b-transparent'
          }`}
        />
      </div>
    </div>
  );
};

export default ChatMessage;