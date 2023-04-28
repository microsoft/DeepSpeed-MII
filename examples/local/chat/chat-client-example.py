# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

# Run `chat-server-example.py` before running this script
generator = mii.mii_query_handle("chat_example_deployment")

# You can set a session name
session_id = "chat_example_session"
# You need to call `create_session` before you start a multi-turn conversation session
generator.create_session(session_id)

queries = ["Hello", "How are you doing?"]
query_template = "Human: {}\n Assistant: "

for q in queries:
    # A session ID is given as a keyword argument
    result = generator.query({"query": query_template.format(q)},
                             session_id=session_id,
                             max_new_tokens=128)
    print(result.response)

# You need to destroy the session after finishing the conversation
generator.destroy_session(session_id)
