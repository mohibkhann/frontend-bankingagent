import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import  ChatMessage  from '@/components/ChatMessage';
import { TypingIndicator } from '@/components/TypingIndicator';
import { FeatureCard } from '@/components/FeatureCard';
import { useToast } from '@/hooks/use-toast';

import { Send, MessageSquare, BarChart3, HelpCircle } from 'lucide-react';
import { useBankingChat } from '@/context/BankingChatContext';

const Index = () => {
  const [inputValue, setInputValue] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);
  const { messages, isTyping, sendMessage, clearChat, error } = useBankingChat();
  const { toast } = useToast();

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  useEffect(() => {
    if (error) {
      toast({
        title: 'Error',
        description: error,
        variant: 'destructive',
      });
    }
  }, [error, toast]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isTyping) return;

    const message = inputValue.trim();
    setInputValue('');
    try {

       const data = await sendMessage(message);
       console.log(message)

       



    } catch (err){
      console.log(err)
    }
    

  };

  const handleQuickAction = (message: string) => {
    if (isTyping) return;
    sendMessage(message);
  };

  const showWelcome = messages.length === 0 && !isTyping;

  return (
    <div className="min-h-screen flex flex-col">
      {/* Enhanced Header */}
      <header className="glass-container rounded-3xl m-6 mb-4 p-8 text-center relative overflow-hidden group">
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-primary via-secondary to-transparent animate-pulse"></div>
        
        {/* Floating particles */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-primary/20 rounded-full animate-float" style={{ animationDelay: '0s', animationDuration: '4s' }}></div>
          <div className="absolute top-3/4 right-1/4 w-1 h-1 bg-secondary/30 rounded-full animate-float" style={{ animationDelay: '2s', animationDuration: '6s' }}></div>
          <div className="absolute top-1/2 right-1/3 w-1.5 h-1.5 bg-accent-blue/25 rounded-full animate-float" style={{ animationDelay: '1s', animationDuration: '5s' }}></div>
        </div>

        <div className="relative z-10">
          <h1 className="gx-logo text-5xl md:text-7xl mb-6 group-hover:scale-105 transition-transform duration-500">GXBank</h1>
          <h2 className="text-xl md:text-3xl font-semibold text-foreground/90 mb-3 tracking-tight">
            AI-Powered Banking Assistant
          </h2>
          <p className="text-muted-foreground text-lg">Multi-Agent Intelligence • Your Financial Partner</p>
          
          {/* Subtle accent line */}
          <div className="mt-6 mx-auto w-24 h-0.5 bg-gradient-to-r from-primary to-secondary rounded-full opacity-60"></div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        <div className="flex-1 max-w-6xl mx-auto w-full px-6 pb-6">
          {/* Chat Container */}
          <Card className="glass-container h-[calc(100vh-300px)] flex flex-col">
            <CardContent className="flex-1 flex flex-col p-0 overflow-hidden">
              {/* Messages Area */}
              <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
                {showWelcome ? (
                  <div className="h-full flex flex-col">
                    {/* Enhanced Welcome Section */}
                    <div className="text-center mb-12 relative">
                      {/* Animated background glow */}
                      <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-secondary/5 to-accent-blue/5 rounded-3xl blur-3xl animate-pulse"></div>
                      
                      <div className="relative z-10">
                        <div className="text-6xl mb-6 animate-float filter drop-shadow-lg" style={{ animationDuration: '4s' }}>
                          🏦
                        </div>
                        <h3 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-accent-green via-primary to-secondary bg-clip-text text-transparent mb-6">
                          Welcome to GX Bank Assistant!
                        </h3>
                        <p className="text-xl text-foreground/80 leading-relaxed max-w-4xl mx-auto font-medium">
                          Your intelligent multi-agent banking companion, powered by advanced AI to help you manage your finances with 
                          <span className="text-primary font-semibold"> precision</span> and 
                          <span className="text-secondary font-semibold"> insight</span>.
                        </p>
                      </div>
                    </div>

                    {/* Enhanced Feature Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
                      <FeatureCard
                        color="blue"
                        title="Smart Spending Analysis"
                        example="How much did I spend on dining last month?"
                        description="Get detailed breakdowns of your spending patterns with intelligent categorization."
                      />
                      <FeatureCard
                        color="purple"
                        title="Financial Insights & Comparisons"
                        example="How do I compare to similar customers?"
                        description="Benchmark your financial habits against peer groups."
                      />
                      <FeatureCard
                        color="green"
                        title="Dynamic Budget Management"
                        example="Create a $800 monthly grocery budget"
                        description="Set up and track budgets with real-time progress monitoring."
                      />
                      <FeatureCard
                        color="orange"
                        title="Personalized Recommendations"
                        example="Which savings account suits me best?"
                        description="Get AI-powered recommendations based on your financial profile."
                      />
                      <FeatureCard
                        color="cyan"
                        title="Banking Products & Services"
                        example="What credit cards do you recommend?"
                        description="Explore our full range of banking products tailored to your needs."
                      />
                      <FeatureCard
                        color="red"
                        title="Predictive Analytics"
                        example="Predict my spending for next month"
                        description="Leverage AI to forecast trends and plan ahead."
                      />
                    </div>

                    {/* Enhanced Pro Tip */}
                    <div className="glass-container rounded-2xl p-8 text-center border-secondary/30 bg-gradient-to-br from-secondary/5 to-primary/5 relative overflow-hidden">
                      <div className="absolute inset-0 bg-gradient-to-r from-secondary/10 via-transparent to-primary/10 animate-pulse"></div>
                      
                      <div className="relative z-10">
                        <div className="inline-flex items-center justify-center w-12 h-12 bg-gradient-to-r from-secondary to-primary rounded-full mb-4">
                          <span className="text-2xl">💡</span>
                        </div>
                        <h4 className="text-secondary font-bold text-xl mb-3">Pro Tip</h4>
                        <p className="text-foreground/90 text-lg leading-relaxed">
                          Ask me anything in natural language - I understand context and can help with complex financial queries!
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {messages.map((message) => (
                      <ChatMessage
                        key={message.id}
                        message = {message}
                      />
                    ))}
                    {isTyping && <TypingIndicator />}
                    <div ref={chatEndRef} />
                  </div>
                )}
              </div>

              {/* Input Area */}
              <div className="p-6 border-t border-border/50">
                <form onSubmit={handleSubmit} className="flex gap-3">
                  <Input
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder="Ask me about your spending, budgets, or banking products..."
                    className="chat-input flex-1"
                    disabled={isTyping}
                  />
                  <Button
                    type="submit"
                    variant="primary"
                    size="lg"
                    disabled={!inputValue.trim() || isTyping}
                    className="px-8"
                  >
                    <Send className="h-5 w-5" />
                  </Button>
                </form>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Action Buttons */}
        <div className="px-6 pb-6">
          <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-4">
            <Button
              variant="glass"
              onClick={() => clearChat()}
              disabled={isTyping}
              className="flex items-center gap-2"
            >
              <MessageSquare className="h-5 w-5" />
              New Conversation
            </Button>
            <Button
              variant="glass"
              onClick={() => handleQuickAction('Show me a comprehensive summary of my spending and financial activity this month')}
              disabled={isTyping}
              className="flex items-center gap-2"
            >
              <BarChart3 className="h-5 w-5" />
              Quick Financial Summary
            </Button>
            <Button
              variant="glass"
              onClick={() => handleQuickAction('Show me help and examples of what I can ask you')}
              disabled={isTyping}
              className="flex items-center gap-2"
            >
              <HelpCircle className="h-5 w-5" />
              Help & Examples
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
