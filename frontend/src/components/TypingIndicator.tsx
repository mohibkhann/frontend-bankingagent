import React from 'react';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';

export const TypingIndicator: React.FC = () => {
  return (
    <div className="flex justify-start mb-6">
      <div className="typing-indicator max-w-80">
        <div className="flex items-center gap-3 mb-2">
          <Avatar className="h-8 w-8">
            <AvatarFallback className="bg-secondary text-secondary-foreground text-sm font-semibold">
              GX
            </AvatarFallback>
          </Avatar>
          <div className="font-bold text-secondary">GX Assistant is analyzing your request...</div>
        </div>
        <div className="typing-dots">
          <div className="typing-dot"></div>
          <div className="typing-dot"></div>
          <div className="typing-dot"></div>
        </div>
      </div>
    </div>
  );
};