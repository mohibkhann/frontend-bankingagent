import React from 'react';

interface FeatureCardProps {
  title: string;
  description: string;
  example: string;
  color: 'blue' | 'purple' | 'green' | 'orange' | 'cyan' | 'red';
}

export const FeatureCard: React.FC<FeatureCardProps> = ({ title, description, example, color }) => {
  const getIcon = () => {
    switch (color) {
      case 'blue': return '📊';
      case 'purple': return '🔮';
      case 'green': return '💰';
      case 'orange': return '🎯';
      case 'cyan': return '🏪';
      case 'red': return '📈';
      default: return '💡';
    }
  };

  return (
    <div className={`feature-card ${color} group`}>
      <div className="flex items-start gap-3 mb-4">
        <div className="text-2xl group-hover:scale-110 transition-transform duration-300">
          {getIcon()}
        </div>
        <div className="flex-1">
          <h3 className="feature-title text-lg font-bold mb-2 leading-tight group-hover:text-opacity-90 transition-all duration-300">
            {title}
          </h3>
        </div>
      </div>
      
      <div className="bg-gradient-to-r from-muted/30 to-muted/10 rounded-lg p-3 mb-4 border-l-2 border-muted-foreground/20">
        <p className="text-muted-foreground text-sm italic font-medium">"{example}"</p>
      </div>
      
      <p className="text-foreground/80 text-sm leading-relaxed group-hover:text-foreground transition-colors duration-300">
        {description}
      </p>
    </div>
  );
};