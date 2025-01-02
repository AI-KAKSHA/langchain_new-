
# LangChain Workflow Examples

This project demonstrates how LangChain can handle workflows with examples like:
1. **Sequential Chains**: Tasks executed in sequence.
2. **Parallel Chains**: Tasks executed in parallel.
3. **Agents with Tools**: Dynamic workflows using tools.
4. **Memory-Enabled Workflows**: Retain context in conversations.

## How to Run
1. Install LangChain and OpenAI libraries:
   ```bash
   pip install langchain openai
   ```
2. Open the `LangChain_Workflows.ipynb` in Jupyter Notebook.
3. Follow each example and execute the code cells.

## Examples Overview
### Sequential Chains
- Define tasks that are executed in a sequential manner, where the output of one task serves as input for the next.

### Parallel Chains
- Execute multiple tasks simultaneously and aggregate their outputs.

### Agents with Tools
- Use dynamic agents that select the appropriate tools based on the task at hand.

### Memory-Enabled Workflows
- Store and use conversation history to maintain context across multiple interactions.

Enjoy experimenting with LangChain workflows!
