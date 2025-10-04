from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class MemoryManager:
    """Manages conversation memory."""
    
    def __init__(self):
        # Initialize chat history
        self.chat_history = InMemoryChatMessageHistory()
    
    def add_message(self, role_or_message, content=None):
        """Add a single message to memory (for compatibility)
        
        Can be called as:
        - add_message(message_object) - where message_object is HumanMessage/AIMessage/SystemMessage
        - add_message("human", "content") - to add a human message
        - add_message("assistant", "content") - to add an AI message
        """

        print(f"Adding message to memory: role_or_message={role_or_message}, content={content}")
        if content is not None:
            # Called with role and content
            # Extract text if content is an object
            content_text = content
            if hasattr(content, 'content'):
                content_text = content.content
            elif hasattr(content, 'text'):
                content_text = content.text
            elif not isinstance(content, str):
                content_text = str(content)
                
            if role_or_message == "human":
                self.chat_history.add_user_message(content_text)
            elif role_or_message in ["assistant", "ai"]:
                self.chat_history.add_ai_message(content_text)
            elif role_or_message == "system":
                self.chat_history.messages.append(SystemMessage(content=content_text))
        elif isinstance(role_or_message, (HumanMessage, AIMessage, SystemMessage)):
            # Called with message object
            self.chat_history.messages.append(role_or_message)
        else:
            # If it's just a string, add as AI message
            self.chat_history.add_ai_message(str(role_or_message))
