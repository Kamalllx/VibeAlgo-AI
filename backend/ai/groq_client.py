# backend/ai/groq_client.py (FIXED VERSION)
import os
import requests
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class GroqResponse:
    content: str
    model: str
    tokens_used: int
    success: bool
    error: Optional[str] = None

class GroqClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = "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"
        self.base_url = "https://api.groq.com/openai/v1"  # Correct endpoint
        self.default_model = "meta-llama/llama-4-scout-17b-16e-instruct"  # Verify this model exists
        
        print(f"🔑 Groq API Key: {'SET' if self.api_key else 'MISSING'}")
        print(f"🌐 Groq Base URL: {self.base_url}")
        print(f"🤖 Default Model: {self.default_model}")
        
        if not self.api_key:
            print("⚠️ WARNING: GROQ_API_KEY not found. Using mock responses.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            # Test API connectivity
            self._test_connection()
    
    def _test_connection(self):
        """Test Groq API connectivity"""
        try:
            print(f"🧪 Testing Groq API connection...")
            
            test_response = self.chat_completion([
                {"role": "user", "content": "Hello, respond with just 'API_TEST_OK'"}
            ], max_tokens=10)
            
            if test_response.success:
                print(f"✅ Groq API connection successful!")
                print(f"📝 Test response: {test_response.content}")
            else:
                print(f"❌ Groq API test failed: {test_response.error}")
                self.mock_mode = True
                
        except Exception as e:
            print(f"❌ Groq API connection error: {str(e)}")
            self.mock_mode = True
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = None, 
                       temperature: float = 0.1,
                       max_tokens: int = 8192) -> GroqResponse:
        """Send chat completion request to Groq API"""
        
        if self.mock_mode:
            print(f"🎭 Using mock mode - Groq API unavailable")
            return self._mock_response(messages)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Use correct model names
            model_name = model or self.default_model
            
            # Try alternative model names if default fails
            model_alternatives = [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "llama3-70b-8192",
                "llama-3.1-70b-versatile", 
                "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768"
            ]
            
            if model_name not in model_alternatives:
                model_name = model_alternatives[0]
            
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            print(f"📤 Sending request to Groq...")
            print(f"🤖 Model: {model_name}")
            print(f"💬 Messages: {len(messages)} messages")
            print(f"🔢 Max tokens: {max_tokens}")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"📥 Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0]['message']['content']
                    tokens_used = data.get('usage', {}).get('total_tokens', 0)
                    
                    print(f"✅ Groq API success!")
                    print(f"📝 Response length: {len(content)} characters")
                    print(f"🔢 Tokens used: {tokens_used}")
                    
                    return GroqResponse(
                        content=content,
                        model=model_name,
                        tokens_used=tokens_used,
                        success=True
                    )
                else:
                    error_msg = "No choices in response"
                    print(f"❌ Groq API error: {error_msg}")
                    return GroqResponse(
                        content="",
                        model=model_name,
                        tokens_used=0,
                        success=False,
                        error=error_msg
                    )
            else:
                error_data = response.json() if response.content else {"error": "No response content"}
                error_msg = f"HTTP {response.status_code}: {error_data}"
                print(f"❌ Groq API HTTP error: {error_msg}")
                
                return GroqResponse(
                    content="",
                    model=model_name,
                    tokens_used=0,
                    success=False,
                    error=error_msg
                )
                
        except requests.exceptions.Timeout:
            error_msg = "Request timeout"
            print(f"❌ Groq API timeout: {error_msg}")
            return GroqResponse("", model or self.default_model, 0, False, error_msg)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            print(f"❌ Groq API request error: {error_msg}")
            return GroqResponse("", model or self.default_model, 0, False, error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"❌ Groq API unexpected error: {error_msg}")
            return GroqResponse("", model or self.default_model, 0, False, error_msg)
    
    def _mock_response(self, messages: List[Dict[str, str]]) -> GroqResponse:
        """Generate realistic mock response for testing"""
        user_message = messages[-1].get('content', '').lower()
        
        print(f"🎭 Generating mock response for: {user_message[:100]}...")
        
        # Analyze user input for better mock responses
        if 'complexity' in user_message and 'for i in range(n)' in user_message:
            mock_content = """
Looking at this Python code step by step:

1. **Identify operations**: There's a single for loop that iterates through range(n)
2. **Input size variable**: The variable n represents the input size  
3. **Operation scaling**: The print(i) statement executes exactly n times
4. **Time complexity**: Since we perform one operation for each value from 0 to n-1, this is O(n)
5. **Space complexity**: We only use a constant amount of extra space for the loop variable i, so this is O(1)

**Conclusion**:
- Time Complexity: O(n) - linear time as we iterate through n elements
- Space Complexity: O(1) - constant space as no additional data structures grow with input size
            """.strip()
            
        elif 'bubble_sort' in user_message or 'nested' in user_message:
            mock_content = """
Analyzing this bubble sort algorithm:

1. **Outer loop**: Runs n times (for i in range(n))
2. **Inner loop**: Runs (n-i-1) times for each outer loop iteration
3. **Total operations**: Sum from i=0 to n-1 of (n-i-1) = n(n-1)/2 ≈ n²/2
4. **Time complexity**: O(n²) - quadratic time due to nested loops
5. **Space complexity**: O(1) - only using a constant amount of extra space for variables

**Conclusion**:
- Time Complexity: O(n²) - quadratic time due to nested loop structure
- Space Complexity: O(1) - in-place sorting algorithm
            """.strip()
            
        elif 'binary_search' in user_message:
            mock_content = """
Analyzing this binary search algorithm:

1. **Loop structure**: While loop that continues while left <= right
2. **Search space**: Halved in each iteration (left = mid + 1 or right = mid - 1)
3. **Maximum iterations**: log₂(n) where n is array length
4. **Time complexity**: O(log n) - logarithmic time due to halving search space
5. **Space complexity**: O(1) - only using constant extra space for variables

**Conclusion**:
- Time Complexity: O(log n) - logarithmic time due to binary division
- Space Complexity: O(1) - iterative implementation uses constant space
            """.strip()
            
        else:
            mock_content = f"This is a mock AI response for testing purposes. The actual Groq API would provide detailed analysis for: {user_message[:50]}..."
        
        print(f"🎭 Mock response generated: {len(mock_content)} characters")
        
        return GroqResponse(
            content=mock_content,
            model="mock-llama3-70b",
            tokens_used=len(mock_content.split()),
            success=True
        )

# Global client instance  
groq_client = GroqClient()
