# AI DATA SCIENCE TEAM
# ***
# * Agents: Feature Engineering Agent

# Libraries
from typing import TypedDict, Annotated, Sequence, Literal
import operator

# Fix matplotlib backend to prevent GUI crashes in threading environments
import matplotlib
matplotlib.use('Agg')

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.types import Command, Checkpointer
from langgraph.checkpoint.memory import MemorySaver

import os
import json
import pandas as pd

from IPython.display import Markdown

from app.templates import(
    node_func_execute_agent_code_on_data, 
    node_func_human_review,
    node_func_fix_agent_code, 
    node_func_report_agent_outputs,
    create_coding_agent_graph,
    BaseAgent,
)
from app.parsers.parsers import PythonOutputParser
from app.utils.regex import (
    relocate_imports_inside_function, 
    add_comments_to_top, 
    format_agent_name, 
    format_recommended_steps, 
    get_generic_summary,
)
from app.tools.dataframe import get_dataframe_summary
from app.utils.logging import log_ai_function

def fix_regex_escaping(code: str) -> str:
    """
    Fix common regex escaping issues in generated code.
    
    This function automatically converts problematic regex patterns
    to use proper Python string escaping or raw strings.
    """
    try:
        import re
        
        # Fix the most common regex escaping issues:
        # 1. Fix string literals with backslash-dot patterns
        # 2. Fix double backslash patterns that got corrupted
        # 3. Fix common regex patterns in pandas str.extract()
        
        # Pattern 1: Fix basic regex patterns with backslash-dot
        pattern1 = r"'([^']*\\\.[^']*?)'"
        def replace_with_raw1(match):
            content = match.group(1)
            return f"r'{content}'"
        
        fixed_code = re.sub(pattern1, replace_with_raw1, code)
        
        # Pattern 2: Handle double quotes too
        pattern2 = r'"([^"]*\\\.[^"]*?)"'
        def replace_with_raw2(match):
            content = match.group(1)
            return f'r"{content}"'
        
        fixed_code = re.sub(pattern2, replace_with_raw2, fixed_code)
        
        # Pattern 3: Specific fix for corrupted raw string prefixes like "rr'"
        # This handles cases where the LLM generates "rr'" instead of "r'"
        fixed_code = re.sub(r'\brr\'([^\']*?)\'', r"r'\1'", fixed_code)
        fixed_code = re.sub(r'\brr"([^"]*?)"', r'r"\1"', fixed_code)
        
        # Pattern 4: Fix common pandas regex patterns specifically
        # Look for .str.extract( followed by non-raw string with backslash patterns
        extract_pattern = r'\.str\.extract\(\s*\'([^\']*\\[^\']*?)\''
        def fix_extract(match):
            content = match.group(1)
            return f".str.extract(r'{content}'"
        
        fixed_code = re.sub(extract_pattern, fix_extract, fixed_code)
        
        # Pattern 5: Same for double quotes in extract
        extract_pattern2 = r'\.str\.extract\(\s*"([^"]*\\[^"]*?)"'
        def fix_extract2(match):
            content = match.group(1)
            return f'.str.extract(r"{content}"'
        
        fixed_code = re.sub(extract_pattern2, fix_extract2, fixed_code)
        
        return fixed_code
        
    except Exception as e:
        # If regex fixing fails, return original code
        print(f"Warning: Could not fix regex escaping: {e}")
        return code

# Setup
AGENT_NAME = "feature_engineering_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Class

class FeatureEngineeringAgent(BaseAgent):
    """
    Creates a feature engineering agent that can process datasets based on user-defined instructions or 
    default feature engineering steps. The agent generates a Python function to engineer features, executes it, 
    and logs the process, including code and errors. It is designed to facilitate reproducible and 
    customizable feature engineering workflows.

    The agent can perform the following default feature engineering steps unless instructed otherwise:
    - Convert features to appropriate data types
    - Remove features that have unique values for each row
    - Remove constant features
    - Encode high-cardinality categoricals (threshold <= 5% of dataset) as 'other'
    - One-hot-encode categorical variables
    - Convert booleans to integer (1/0)
    - Create datetime-based features (if applicable)
    - Handle target variable encoding if specified
    - Any user-provided instructions to add, remove, or modify steps

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the feature engineering function.
    n_samples : int, optional
        Number of samples used when summarizing the dataset. Defaults to 30.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the file for saving the generated response. Defaults to "feature_engineer.py".
    function_name : str, optional
        Name of the function for data visualization. Defaults to "feature_engineer".
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. If False, a unique file name is created. Defaults to True.
    human_in_the_loop : bool, optional
        Enables user review of feature engineering instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        If True, skips the default recommended steps. Defaults to False.
    bypass_explain_code : bool, optional
        If True, skips the step that provides code explanations. Defaults to False.
    checkpointer : Checkpointer, optional
        Checkpointer to save and load the agent's state. Defaults to None.

    Methods
    -------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled state graph.
    ainvoke_agent(
        user_instructions: str, 
        data_raw: pd.DataFrame, 
        target_variable: str = None, 
        max_retries=3, 
        retry_count=0
    )
        Engineers features from the provided dataset asynchronously based on user instructions.
    invoke_agent(
        user_instructions: str, 
        data_raw: pd.DataFrame, 
        target_variable: str = None, 
        max_retries=3, 
        retry_count=0
    )
        Engineers features from the provided dataset synchronously based on user instructions.
    get_workflow_summary()
        Retrieves a summary of the agent's workflow.
    get_log_summary()
        Retrieves a summary of logged operations if logging is enabled.
    get_data_engineered()
        Retrieves the feature-engineered dataset as a pandas DataFrame.
    get_data_raw()
        Retrieves the raw dataset as a pandas DataFrame.
    get_feature_engineer_function()
        Retrieves the generated Python function used for feature engineering.
    get_recommended_feature_engineering_steps()
        Retrieves the agent's recommended feature engineering steps.
    get_response()
        Returns the response from the agent as a dictionary.
    show()
        Displays the agent's mermaid diagram.

    Examples
    --------
    ```python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from app.agents import FeatureEngineeringAgent

    llm = ChatOpenAI(model="gpt-4o-mini")

    feature_agent = FeatureEngineeringAgent(
        model=llm, 
        n_samples=30, 
        log=True, 
        log_path="logs", 
        human_in_the_loop=True
    )

    df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")

    feature_agent.invoke_agent(
        user_instructions="Also encode the 'PaymentMethod' column with one-hot encoding.", 
        data_raw=df, 
        target_variable="Churn",
        max_retries=3,
        retry_count=0
    )

    engineered_data = feature_agent.get_data_engineered()
    response = feature_agent.get_response()
    ```
    
    Returns
    -------
    FeatureEngineeringAgent : langchain.graphs.CompiledStateGraph 
        A feature engineering agent implemented as a compiled state graph.
    """

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="feature_engineer.py",
        function_name="feature_engineer",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        checkpointer=None,
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "bypass_recommended_steps": bypass_recommended_steps,
            "bypass_explain_code": bypass_explain_code,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """
        Create the compiled graph for the feature engineering agent. 
        Running this method will reset the response to None.
        """
        self.response = None
        return make_feature_engineering_agent(**self._params)

    def update_params(self, **kwargs):
        """
        Updates the agent's parameters and rebuilds the compiled graph.
        """
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(
        self, 
        data_raw: pd.DataFrame, 
        user_instructions: str=None, 
        target_variable: str = None, 
        max_retries=3, 
        retry_count=0,
        **kwargs
    ):
        """
        Asynchronously engineers features for the provided dataset.
        The response is stored in the 'response' attribute.

        Parameters
        ----------
        data_raw : pd.DataFrame
            The raw dataset to be processed.
        user_instructions : str, optional
            Instructions for feature engineering.
        target_variable : str, optional
            The name of the target variable (if any).
        max_retries : int
            Maximum retry attempts.
        retry_count : int
            Current retry attempt count.
        **kwargs
            Additional keyword arguments to pass to ainvoke().

        Returns
        -------
        None
        """
        response = await self._compiled_graph.ainvoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "target_variable": target_variable,
            "max_retries": max_retries,
            "retry_count": retry_count
        }, **kwargs)
        self.response = response
        return None

    def invoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str=None,
        target_variable: str = None,
        max_retries=3,
        retry_count=0,
        **kwargs
    ):
        """
        Synchronously engineers features for the provided dataset.
        The response is stored in the 'response' attribute.

        Parameters
        ----------
        data_raw : pd.DataFrame
            The raw dataset to be processed.
        user_instructions : str
            Instructions for feature engineering agent.
        target_variable : str, optional
            The name of the target variable (if any).
        max_retries : int
            Maximum retry attempts.
        retry_count : int
            Current retry attempt count.
        **kwargs
            Additional keyword arguments to pass to invoke().

        Returns
        -------
        None
        """
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "target_variable": target_variable,
            "max_retries": max_retries,
            "retry_count": retry_count
        }, **kwargs)
        self.response = response
        return None

    def get_workflow_summary(self, markdown=False):
        """
        Retrieves the agent's workflow summary, if logging is enabled.
        """
        if self.response and self.response.get("messages"):
            summary = get_generic_summary(json.loads(self.response.get("messages")[-1].content))
            if markdown:
                return Markdown(summary)
            else:
                return summary

    def get_log_summary(self, markdown=False):
        """
        Logs a summary of the agent's operations, if logging is enabled.
        """
        if self.response:
            if self.response.get('feature_engineer_function_path'):
                log_details = f"""
## Featuring Engineering Agent Log Summary:

Function Path: {self.response.get('feature_engineer_function_path')}

Function Name: {self.response.get('feature_engineer_function_name')}
                """
                if markdown:
                    return Markdown(log_details) 
                else:
                    return log_details

    def get_data_engineered(self):
        """
        Retrieves the engineered data stored after running invoke/ainvoke.

        Returns
        -------
        pd.DataFrame or None
            The engineered dataset as a pandas DataFrame.
        """
        if self.response and "data_engineered" in self.response:
            return pd.DataFrame(self.response["data_engineered"])
        return None

    def get_data_raw(self):
        """
        Retrieves the raw data.

        Returns
        -------
        pd.DataFrame or None
            The raw dataset as a pandas DataFrame if available.
        """
        if self.response and "data_raw" in self.response:
            return pd.DataFrame(self.response["data_raw"])
        return None

    def get_feature_engineer_function(self, markdown=False):
        """
        Retrieves the feature engineering function generated by the agent.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns the function in Markdown code block format.

        Returns
        -------
        str or None
            The Python function code, or None if unavailable.
        """
        if self.response and "feature_engineer_function" in self.response:
            code = self.response["feature_engineer_function"]
            if markdown:
                return Markdown(f"```python\n{code}\n```")
            return code
        return None

    def get_recommended_feature_engineering_steps(self, markdown=False):
        """
        Retrieves the agent's recommended feature engineering steps.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns the steps in Markdown format.

        Returns
        -------
        str or None
            The recommended steps, or None if not available.
        """
        if self.response and "recommended_steps" in self.response:
            steps = self.response["recommended_steps"]
            if markdown:
                return Markdown(steps)
            return steps
        return None

    


# * Feature Engineering Agent

def make_feature_engineering_agent(
    model, 
    n_samples=30,
    log=False, 
    log_path=None, 
    file_name="feature_engineer.py",
    function_name="feature_engineer",
    overwrite = True, 
    human_in_the_loop=False, 
    bypass_recommended_steps=False, 
    bypass_explain_code=False,
    checkpointer=None,
):
    """
    Creates a feature engineering agent that can be run on a dataset. The agent applies various feature engineering
    techniques, such as encoding categorical variables, scaling numeric variables, creating interaction terms,
    and generating polynomial features. The agent takes in a dataset and user instructions and outputs a Python
    function for feature engineering. It also logs the code generated and any errors that occur.
    
    The agent is instructed to apply the following feature engineering techniques:
    
    - Remove string or categorical features with unique values equal to the size of the dataset
    - Remove constant features with the same value in all rows
    - High cardinality categorical features should be encoded by a threshold <= 5 percent of the dataset, by converting infrequent values to "other"
    - Encoding categorical variables using OneHotEncoding
    - Numeric features should be left untransformed
    - Create datetime-based features if datetime columns are present
    - If a target variable is provided:
        - If a categorical target variable is provided, encode it using LabelEncoding
        - All other target variables should be converted to numeric and unscaled
    - Convert any boolean True/False values to 1/0
    - Return a single data frame containing the transformed features and target variable, if one is provided.
    - Any specific instructions provided by the user

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use to generate code.
    n_samples : int, optional
        The number of data samples to use for generating the feature engineering code. Defaults to 30.
        If you get an error due to maximum tokens, try reducing this number.
        > "This model's maximum context length is 128000 tokens. However, your messages resulted in 333858 tokens. Please reduce the length of the messages."
    log : bool, optional
        Whether or not to log the code generated and any errors that occur.
        Defaults to False.
    log_path : str, optional
        The path to the directory where the log files should be stored. Defaults to "logs/".
    file_name : str, optional
        The name of the file to save the log to. Defaults to "feature_engineer.py".
    function_name : str, optional
        The name of the function that will be generated. Defaults to "feature_engineer".
    overwrite : bool, optional
        Whether or not to overwrite the log file if it already exists. If False, a unique file name will be created. 
        Defaults to True.
    human_in_the_loop : bool, optional
        Whether or not to use human in the loop. If True, adds an interput and human in the loop step that asks the user to review the feature engineering instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        Bypass the recommendation step, by default False
    bypass_explain_code : bool, optional
        Bypass the code explanation step, by default False.
    checkpointer : Checkpointer, optional
        Checkpointer to save and load the agent's state. Defaults to None.

    Examples
    -------
    ``` python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import feature_engineering_agent

    llm = ChatOpenAI(model="gpt-4o-mini")

    feature_engineering_agent = make_feature_engineering_agent(llm)

    df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")

    response = feature_engineering_agent.invoke({
        "user_instructions": None,
        "target_variable": "Churn",
        "data_raw": df.to_dict(),
        "max_retries": 3,
        "retry_count": 0
    })

    pd.DataFrame(response['data_engineered'])
    ```

    Returns
    -------
    app : langchain.graphs.CompiledStateGraph
        The feature engineering agent as a state graph.
    """
    llm = model
    
    if human_in_the_loop:
        if checkpointer is None:
            print("Human in the loop is enabled. A checkpointer is required. Setting to MemorySaver().")
            checkpointer = MemorySaver()
    
    # Human in th loop requires recommended steps
    if bypass_recommended_steps and human_in_the_loop:
        bypass_recommended_steps = False
        print("Bypass recommended steps set to False to enable human in the loop.")

    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = "logs/"
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        data_raw: dict
        data_engineered: dict
        target_variable: str
        all_datasets_summary: str
        feature_engineer_function: str
        feature_engineer_function_path: str
        feature_engineer_file_name: str
        feature_engineer_function_name: str
        feature_engineer_error: str
        max_retries: int
        retry_count: int

    def recommend_feature_engineering_steps(state: GraphState):
        """
        Recommend a series of feature engineering steps based on the input data.
        These recommended steps will be appended to the user_instructions.
        """
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND FEATURE ENGINEERING STEPS")

        # Prompt to get recommended steps from the LLM
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a Feature Engineering Expert. Given the following information about the data, 
            recommend a series of numbered steps to take to engineer features. 
            The steps should be tailored to the data characteristics and should be helpful 
            for a feature engineering agent that will be implemented.
            
            General Steps:
            Things that should be considered in the feature engineering steps:
            
            * Convert features to the appropriate data types based on their sample data values
            * Remove string or categorical features with unique values equal to the size of the dataset
            * Remove constant features with the same value in all rows
            * High cardinality categorical features should be encoded by a threshold <= 5 percent of the dataset, by converting infrequent values to "other"
            * Encoding categorical variables using OneHotEncoding
            * Numeric features should be left untransformed
            * Create datetime-based features if datetime columns are present
            * If a target variable is provided:
                * If a categorical target variable is provided, encode it using LabelEncoding
                * All other target variables should be converted to numeric and unscaled
            * Convert any Boolean (True/False) values to integer (1/0) values. This should be performed after one-hot encoding.
            
            Custom Steps:
            * Analyze the data to determine if any additional feature engineering steps are needed.
            * Recommend steps that are specific to the data provided. Include why these steps are necessary or beneficial.
            * If no additional steps are needed, simply state that no additional steps are required.
            
            IMPORTANT:
            Make sure to take into account any additional user instructions that may add, remove or modify some of these steps. Include comments in your code to explain your reasoning for each step. Include comments if something is not done because a user requested. Include comments if something is done because a user requested.
            
            User instructions:
            {user_instructions}
            
            Previously Recommended Steps (if any):
            {recommended_steps}
            
            Below are summaries of all datasets provided:
            {all_datasets_summary}

            Return steps as a numbered list. You can return short code snippets to demonstrate actions. But do not return a fully coded solution. The code will be generated separately by a Coding Agent.
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated user instructions that are not related to the feature engineering.
            """,
            input_variables=["user_instructions", "recommended_steps", "all_datasets_summary"]
        )

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        
        all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples)
        
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)

        steps_agent = recommend_steps_prompt | llm
        recommended_steps = steps_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "recommended_steps": state.get("recommended_steps"),
            "all_datasets_summary": all_datasets_summary_str
        }) 
        
        return {
            "recommended_steps": format_recommended_steps(recommended_steps.content.strip(), heading="# Recommended Feature Engineering Steps:"),
            "all_datasets_summary": all_datasets_summary_str
        }
    
    # Human Review   
    
    prompt_text_human_review = "Are the following feature engineering instructions correct? (Answer 'yes' or provide modifications)\n{steps}"
    
    if not bypass_explain_code:
        def human_review(state: GraphState) -> Command[Literal["recommend_feature_engineering_steps", "explain_feature_engineering_code"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto= 'explain_feature_engineering_code',
                no_goto="recommend_feature_engineering_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="feature_engineer_function",
            )
    else:
        def human_review(state: GraphState) -> Command[Literal["recommend_feature_engineering_steps", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto= '__end__',
                no_goto="recommend_feature_engineering_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="feature_engineer_function", 
            )
    
    def create_feature_engineering_code(state: GraphState):
        if bypass_recommended_steps:
            print(format_agent_name(AGENT_NAME))
            
            data_raw = state.get("data_raw")
            df = pd.DataFrame.from_dict(data_raw)
            
            all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples)
            
            all_datasets_summary_str = "\n\n".join(all_datasets_summary)
            
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")
            
        print("    * CREATE FEATURE ENGINEERING CODE")

        feature_engineering_prompt = PromptTemplate(
            template="""
            You are a Feature Engineering Agent. Your job is to create a {function_name}() function that can be run on the data provided using the following recommended steps.
            
            Recommended Steps:
            {recommended_steps}
            
            Use this information about the data to help determine how to feature engineer the data:
            
            Target Variable (if provided): {target_variable}
            
            Below are summaries of all datasets provided. Use this information about the data to help determine how to feature engineer the data:
            {all_datasets_summary}
            
            You can use Pandas, Numpy, and Scikit Learn libraries to feature engineer the data.
            
            Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), including all imports inside the function.

            Return code to provide the feature engineering function:
            
            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                ...
                
                # CRITICAL: Validate target column exists
                target_col = '{target_variable}'
                if target_col and target_col not in data_engineered.columns:
                    if target_col in data_raw.columns:
                        data_engineered[target_col] = data_raw[target_col]
                    else:
                        print(f"WARNING: Target column {{{{target_col}}}} not found!")
                
                return data_engineered
            
            Best Practices and Error Preventions:
            - Handle missing values in numeric and categorical features before transformations.
            - Avoid creating highly correlated features unless explicitly instructed.
            - Convert Boolean to integer values (0/1) after one-hot encoding unless otherwise instructed.
            
            CRITICAL - Target Column Preservation:
            - NEVER remove or modify the target column: {target_variable}
            - Always check that the target column exists in the final output
            - If target column is missing, add it back from the original data
            - Use unique column names to avoid pandas dropping duplicates
            - Validate final DataFrame has all required columns before returning
            
            IMPORTANT - Use Modern Sklearn Syntax (v1.6+):
            - Use OneHotEncoder(sparse_output=False) instead of OneHotEncoder(sparse=False)
            - Use encoder.get_feature_names_out() instead of encoder.get_feature_names()
            - Use dtype=np.int32 instead of dtype=np.int
            - Always import what you need: from sklearn.preprocessing import OneHotEncoder
            
            CRITICAL - Python Syntax Validation:
            - ALWAYS validate parentheses, brackets, and braces are properly matched
            - Use raw strings for regex patterns: r'pattern' instead of 'pattern'
            - For literal dots in regex, use r'\\.' or '\\\\.'
            - Example: data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)')
            - Double-check all bracket pairs: [], {{}}, ()
            - Test: encoder.fit_transform(data[['column']]) NOT encoder.fit_transform(data[['column'])
            - MANDATORY: Count opening and closing brackets before generating code
            - Avoid unescaped backslashes in string literals
            - Use f-strings or .format() for string formatting, not % formatting
            
            🚨 CRITICAL - Categorical Column Handling:
            MUST handle pandas categorical columns safely to avoid "Cannot setitem on a Categorical with a new category" errors:
            
            ```python
            # SAFE categorical handling for grouping low-frequency values
            for col in categorical_columns:
                if str(data[col].dtype) == 'category':
                    # Method 1: Convert to object first, then group
                    data[col] = data[col].astype('object')
                    
                    # Now safe to assign new values like 'Other'
                    value_counts = data[col].value_counts()
                    low_freq_values = value_counts[value_counts < threshold].index
                    data[col] = data[col].replace(low_freq_values, 'Other')
                
                elif data[col].dtype == 'object':
                    # Standard string/object columns - safe to assign directly
                    value_counts = data[col].value_counts()
                    low_freq_values = value_counts[value_counts < threshold].index
                    data[col] = data[col].replace(low_freq_values, 'Other')
            ```
            
            Alternative method - Add category first:
            ```python
            # Method 2: Add new category then assign
            if str(data[col].dtype) == 'category':
                if 'Other' not in data[col].cat.categories:
                    data[col] = data[col].cat.add_categories(['Other'])
                # Now safe to assign 'Other'
                data[col] = data[col].replace(low_freq_values, 'Other')
            ```
            
            Avoid the following errors:
            
            - name 'OneHotEncoder' is not defined
            - OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'
            - 'OneHotEncoder' object has no attribute 'get_feature_names'
            - argument of type 'method' is not iterable
            - invalid escape sequence '\\.' (use raw strings: r'pattern')
            - DeprecationWarning: invalid escape sequence (use r'' strings for regex)
            - Shape of passed values is (7043, 48), indices imply (7043, 47)
            - name 'numeric_features' is not defined
            - name 'categorical_features' is not defined
            - Cannot setitem on a Categorical with a new category (Other), set the categories first


            """,
            input_variables=["recommended_steps", "target_variable", "all_datasets_summary", "function_name"]
        )

        feature_engineering_agent = feature_engineering_prompt | llm | PythonOutputParser()

        response = feature_engineering_agent.invoke({
            "recommended_steps": state.get("recommended_steps"),
            "target_variable": state.get("target_variable"),
            "all_datasets_summary": all_datasets_summary_str,
            "function_name": function_name
        })
        
        response = relocate_imports_inside_function(response)
        response = fix_regex_escaping(response)
        response = add_comments_to_top(response, agent_name=AGENT_NAME)

        # For logging: store the code generated
        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite
        )

        return {
            "feature_engineer_function": response,
            "feature_engineer_function_path": file_path,
            "feature_engineer_file_name": file_name_2,
            "feature_engineer_function_name": function_name,
            "all_datasets_summary": all_datasets_summary_str
        }

    def execute_feature_engineering_code(state):
        def safe_post_processing(df):
            """Safe post-processing that preserves target column and handles duplicates"""
            if not isinstance(df, pd.DataFrame):
                return df
            
            target_variable = state.get("target_variable")
            
            # Check for duplicate columns
            if df.columns.duplicated().any():
                print(f"⚠️  WARNING: Duplicate columns detected: {df.columns[df.columns.duplicated()].tolist()}")
                
                # Handle duplicates by keeping only the first occurrence
                df = df.loc[:, ~df.columns.duplicated()]
                print(f"✅ Removed duplicate columns. New shape: {df.shape}")
            
            # CRITICAL: Check if feature engineering failed completely
            original_data = state.get("data_raw")
            if original_data and isinstance(original_data, dict):
                original_df = pd.DataFrame.from_dict(original_data)
                original_feature_count = len([col for col in original_df.columns if col != target_variable])
                current_feature_count = len([col for col in df.columns if col != target_variable])
                
                print(f"📊 Original features: {original_feature_count}, Current features: {current_feature_count}")
                
                # If we lost too many features, this indicates failure
                if current_feature_count == 0 and original_feature_count > 0:
                    print(f"🚨 CRITICAL ERROR: All features were dropped! Feature engineering failed.")
                    print(f"🔧 RECOVERY: Using original data with basic preprocessing")
                    
                    # Use original data with basic one-hot encoding
                    df_recovered = original_df.copy()
                    
                    # Apply basic feature engineering
                    categorical_cols = df_recovered.select_dtypes(include=['object']).columns.tolist()
                    if target_variable in categorical_cols:
                        categorical_cols.remove(target_variable)
                    
                    if categorical_cols:
                        print(f"🔧 Applying basic one-hot encoding to: {categorical_cols}")
                        df_recovered = pd.get_dummies(df_recovered, columns=categorical_cols, drop_first=True)
                    
                    df = df_recovered
                    print(f"✅ RECOVERY SUCCESS: {df.shape} with {len([col for col in df.columns if col != target_variable])} features")
            
            # CRITICAL: Ensure target column exists if specified
            if target_variable and target_variable not in df.columns:
                print(f"🚨 CRITICAL ERROR: Target column '{target_variable}' missing from feature engineered data!")
                print(f"   Available columns: {list(df.columns)}")
                
                # Try to recover by loading original data
                try:
                    if original_data and isinstance(original_data, dict):
                        original_df = pd.DataFrame.from_dict(original_data)
                        if target_variable in original_df.columns:
                            print(f"🔧 RECOVERY: Adding target column '{target_variable}' from original data")
                            df[target_variable] = original_df[target_variable]
                        else:
                            print(f"❌ RECOVERY FAILED: Target '{target_variable}' not in original data either")
                except Exception as e:
                    print(f"❌ RECOVERY FAILED: {e}")
            
            print(f"✅ Final feature engineered data shape: {df.shape}")
            print(f"✅ Final columns: {list(df.columns)}")
            
            return df.to_dict()
        
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="data_engineered",
            error_key="feature_engineer_error",
            code_snippet_key="feature_engineer_function",
            agent_function_name=state.get("feature_engineer_function_name"),
            pre_processing=lambda data: pd.DataFrame.from_dict(data),
            post_processing=safe_post_processing,
            error_message_prefix="An error occurred during feature engineering: "
        )

    def fix_feature_engineering_code(state: GraphState):
        feature_engineer_prompt = """
        You are a Feature Engineering Agent. Your job is to fix the {function_name}() function that currently contains errors.
        
        Provide only the corrected function definition for {function_name}().
        
        Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that includes all imports inside the function.
        
        IMPORTANT - Use Modern Sklearn Syntax (v1.6+):
        - Use OneHotEncoder(sparse_output=False) instead of OneHotEncoder(sparse=False)
        - Use encoder.get_feature_names_out() instead of encoder.get_feature_names()
        - Use dtype=np.int32 instead of dtype=np.int
        - Always import what you need: from sklearn.preprocessing import OneHotEncoder
        
        CRITICAL - Python String & Regex Syntax:
        - Use raw strings for regex patterns: r'pattern' instead of 'pattern'
        - For literal dots in regex, use r'\\.' or '\\\\.'
        - Example: data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\\.', expand=False)
        - Avoid unescaped backslashes in string literals
        - Never use prefixes like "rr'" - use only "r'" for raw strings
        
        This is the broken code (please fix): 
        {code_snippet}

        Last Known Error:
        {error}
        """

        # Call the standard fix function but also apply regex escaping
        result = node_func_fix_agent_code(
            state=state,
            code_snippet_key="feature_engineer_function",
            error_key="feature_engineer_error",
            llm=llm,
            prompt_template=feature_engineer_prompt,
            agent_name=AGENT_NAME,
            log=log,
            file_path=state.get("feature_engineer_function_path"),
            function_name=state.get("feature_engineer_function_name"),
        )
        
        # Apply regex escaping fix to the corrected code
        if "feature_engineer_function" in result:
            result["feature_engineer_function"] = fix_regex_escaping(result["feature_engineer_function"])
        
        return result

    # Final reporting node
    def report_agent_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "recommended_steps",
                "feature_engineer_function",
                "feature_engineer_function_path",
                "feature_engineer_function_name",
                "feature_engineer_error",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="Feature Engineering Agent Outputs"
        )
    
    # Create the graph
    node_functions = {
        "recommend_feature_engineering_steps": recommend_feature_engineering_steps,
        "human_review": human_review,
        "create_feature_engineering_code": create_feature_engineering_code,
        "execute_feature_engineering_code": execute_feature_engineering_code,
        "fix_feature_engineering_code": fix_feature_engineering_code,
        "report_agent_outputs": report_agent_outputs,
    }
    
    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="recommend_feature_engineering_steps",
        create_code_node_name="create_feature_engineering_code",
        execute_code_node_name="execute_feature_engineering_code",
        fix_code_node_name="fix_feature_engineering_code",
        explain_code_node_name="report_agent_outputs",
        error_key="feature_engineer_error",
        max_retries_key = "max_retries",
        retry_count_key = "retry_count",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name="human_review",
        checkpointer=checkpointer,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
        agent_name=AGENT_NAME,
    )

    return app
