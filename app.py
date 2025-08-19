from flask import Flask, request, jsonify
from multi_agent_langsmith import graph
from langchain_core.messages import HumanMessage
import uuid

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400

        # Extract user message
        user_message = data["message"]

        # Generate unique thread ID per conversation
        thread_id = data.get("thread_id", str(uuid.uuid4()))
        user_id = data.get("user_id", "anonymous")

        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }

        # Invoke the LangGraph agent
        result = graph.invoke({
            "messages": [HumanMessage(content=user_message)]
        }, config=config)

        # Extract response messages
        responses = [
            {"type": type(msg).__name__, "content": msg.content}
            for msg in result["messages"]
        ]

        return jsonify({
            "thread_id": thread_id,
            "user_id": user_id,
            "responses": responses
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
