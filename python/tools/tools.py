#all the neccessary imports
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import sys
from pathlib import Path
# Import the DataStore 
# from banking_agent.data_store.data_store import DataStore
from typing import Any, Dict, List, Optional, TypedDict, Annotated

from data_store.data_store import DataStore

# for current time calculation 
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Any, Dict, Optional


load_dotenv()


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir
        )
    )
)

def get_current_date_context() -> str:
    """Generate current date context for LLM prompts."""
    now = datetime.now()

    # Calculate relevant dates
    last_month_start = (now - relativedelta(months=1)).replace(day=1)
    last_month_end = now.replace(day=1) - timedelta(days=1)
    current_month_start = now.replace(day=1)
    last_year_start = (now.replace(month=1, day=1) - relativedelta(years=1))
    last_year_end = now.replace(month=1, day=1) - timedelta(days=1)

    context = f"""
CURRENT DATE CONTEXT:

- Today's date: {now.strftime("%Y-%m-%d")}
- Current year: {now.year}
- Current month: {now.strftime("%Y-%m")}
- Last month: {last_month_start.strftime("%Y-%m-%d")} to {last_month_end.strftime("%Y-%m-%d")}
- Current month so far: {current_month_start.strftime("%Y-%m-%d")} to {now.strftime("%Y-%m-%d")}
- Last year: {last_year_start.strftime("%Y-%m-%d")} to {last_year_end.strftime("%Y-%m-%d")}

WHEN USER SAYS:

- "last month" → use dates {last_month_start.strftime("%Y-%m-%d")} to {last_month_end.strftime("%Y-%m-%d")}
- "this month" → use dates {current_month_start.strftime("%Y-%m-%d")} to {now.strftime("%Y-%m-%d")}
- "last year" → use dates {last_year_start.strftime("%Y-%m-%d")} to {last_year_end.strftime("%Y-%m-%d")}
- "recent" or "lately" → use last 30 days: {(now - timedelta(days=30)).strftime("%Y-%m-%d")} to {now.strftime("%Y-%m-%d")}
"""
    return context



# Global DataStore instance for tools
_datastore: Optional[DataStore] = None


def _ensure_datastore() -> DataStore:
    """Ensure global datastore is initialized."""
    global _datastore
    if _datastore is None:
        _datastore = DataStore(
            client_csv_path=Path(__file__).parent / "Banking_Data.csv",
            overall_csv_path=Path(__file__).parent / "overall_data.csv",
            db_path="/Users/mohibalikhan/Desktop/frontend_banking/python/data/banking_data.db",
        )
    return _datastore


@tool
def generate_sql_for_client_analysis(
    user_query: str,
    client_id: int
) -> Dict[str, Any]:
    """
    Generate optimized SQL for client_transactions with current date context and enhanced MCC analysis.
    """
    ds = _ensure_datastore()
    schema = ds.get_schema_info()["client_transactions"]

    # Get current date context
    date_context = get_current_date_context()

    # Build enhanced schema description with MCC intelligence
    schema_desc_lines = [
        "CLIENT_TRANSACTIONS TABLE:",
        f"Description: {schema['description']}",
        "",
        "KEY COLUMNS:"
    ]
    important_columns = {
        "client_id": "INTEGER - Unique client identifier (ALWAYS required in WHERE clause)",
        "date": "TEXT (YYYY-MM-DD) - Transaction date",
        "amount": "REAL - Transaction amount in dollars",
        "mcc_category": "TEXT - High-level spending category (Restaurants, Groceries, Transportation, etc.)",
        "mcc_original": "TEXT - Detailed merchant category (e.g., 'Eating Places and Restaurants', 'Fast Food Restaurants')",
        "mcc_number": "INTEGER - Specific MCC code (e.g., 5812 = Restaurants, 5814 = Fast Food)",
        "merchant_city": "TEXT - City where transaction occurred",
        "is_weekend": "BOOLEAN - Weekend flag",
        "is_night_txn": "BOOLEAN - Night transaction flag",
        "current_age": "INTEGER - Customer age",
        "gender": "TEXT - Customer gender",
        "yearly_income": "REAL - Annual income"
    }
    for col, desc in important_columns.items():
        schema_desc_lines.append(f"- {col}: {desc}")
    schema_desc = "\n".join(schema_desc_lines)

    prompt = ChatPromptTemplate.from_messages([
                ("system", f"""
            You are an expert SQL generator for banking transaction analysis with intelligent MCC (Merchant Category Code) analysis.
            Generate efficient SQLite queries against client_transactions with smart MCC granularity detection.

            {date_context}

            {schema_desc}

            MCC ANALYSIS INTELLIGENCE:
            
            **MCC GRANULARITY LEVELS:**
            1. **HIGH-LEVEL** (mcc_category): Use for general spending analysis
               - Examples: "Restaurants", "Groceries", "Transportation"
               - Query: GROUP BY mcc_category
            
            2. **DETAILED** (mcc_category + mcc_original): Use for specific breakdowns
               - Examples: "Fast Food Restaurants" vs "Eating Places and Restaurants"
               - Query: GROUP BY mcc_category, mcc_original
            
            3. **GRANULAR** (include mcc_number): Use for exact MCC code analysis
               - Examples: MCC 5812 vs MCC 5814
               - Query: SELECT mcc_number, mcc_original, mcc_category

            **WHEN TO USE DETAILED MCC ANALYSIS:**
            - User asks: "breakdown", "detailed", "specific", "what exactly", "what types"
            - User asks: "what restaurants exactly", "break down my grocery spending"
            - Follow-up questions after showing category totals
            - User mentions: "merchant types", "specific categories", "drill down"

            **WHEN TO USE HIGH-LEVEL MCC:**
            - User asks: "total by category", "spending categories", "overall breakdown"
            - First-time category analysis
            - General spending summaries

            **SQL EXAMPLES:**

            High-level category query:
            SELECT mcc_category, SUM(amount) as total_spent, COUNT(*) as transaction_count
            FROM client_transactions 
            WHERE client_id = {client_id}
            GROUP BY mcc_category
            ORDER BY total_spent DESC;

            Detailed breakdown query:
            SELECT mcc_category, mcc_original, SUM(amount) as total_spent, COUNT(*) as transaction_count
            FROM client_transactions 
            WHERE client_id = {client_id}
            GROUP BY mcc_category, mcc_original
            ORDER BY mcc_category, total_spent DESC;

            Granular MCC analysis:
            SELECT mcc_number, mcc_original, mcc_category, SUM(amount) as total_spent
            FROM client_transactions 
            WHERE client_id = {client_id}
            GROUP BY mcc_number, mcc_original, mcc_category
            ORDER BY total_spent DESC;

            Restaurant-specific detailed breakdown:
            SELECT mcc_original, SUM(amount) as total_spent, COUNT(*) as transaction_count
            FROM client_transactions 
            WHERE client_id = {client_id} AND mcc_category = 'Restaurants'
            GROUP BY mcc_original
            ORDER BY total_spent DESC;

            CRITICAL REQUIREMENTS:
            - ALWAYS include WHERE client_id = {client_id}
            - Use the CURRENT DATE CONTEXT above for date filtering
            - If the user explicitly mentions specific dates/years (e.g., "2021 or 2023"), you MAY use those exact periods with range filters (YYYY-01-01 to YYYY-12-31). Otherwise, NEVER hardcode years; derive from the provided date context
            - Use indexed columns: client_id, date, mcc_category, amount
            - Date format YYYY-MM-DD
            - Use meaningful aliases in your query
            - Aggregates: SUM, AVG, COUNT
            - NOT SELF-DECIDING: never pick a "winner" or collapse to a single answer in SQL
            - DO NOT use ORDER BY ... LIMIT 1 to select the max/min period
            - DO NOT use CASE expressions that compare aggregates to choose a label
            - DO NOT use subqueries that filter to only the max/min aggregate
            - INSTEAD return one row per candidate period/category with relevant aggregates
            - Prefer sargable range filters over strftime when possible to leverage indexes
            - RETURN ONLY THE SQL QUERY - no explanations or markdown

            **MCC QUERY DECISION LOGIC:**
            1. Analyze user query for granularity keywords
            2. If detailed analysis requested → include mcc_original
            3. If granular analysis requested → include mcc_number
            4. If general category analysis → use mcc_category only
            5. Always provide rich comparative data without self-deciding

            Generate ONLY the SQL query for:
            """),
                ("human", "{user_query}")
            ])

    try:
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),            
            temperature=0,
        )   
   
        resp = llm.invoke(prompt.format_messages(user_query=user_query))
        
        if not resp or not resp.content:
            return {"error": "No response from LLM"}
            
        sql = resp.content.strip()

        # Strip any fencing or extra text
        lines = [l.strip() for l in sql.splitlines()]
        sql_lines, found = [], False
        for line in lines:
            if line.upper().startswith("SELECT") or found:
                found = True
                sql_lines.append(line)
        sql = "\n".join(sql_lines).strip("```sql ").strip("```")

        # Fallback: extract first SELECT
        if not sql.upper().startswith("SELECT"):
            idx = resp.content.upper().find("SELECT")
            if idx >= 0:
                sql = resp.content[idx:].split("\n\n")[0].strip()
            else:
                return {"error": f"No valid SQL query generated for: {user_query}"}

        # Validate we have a meaningful SQL query
        if not sql or len(sql.strip()) < 10:
            return {"error": f"Generated SQL too short or empty: {sql}"}

        # Validate client_id presence
        if f"client_id = {client_id}" not in sql:
            return {"error": f"Missing client_id filter: {sql}"}

        # Check for hardcoded 2023 dates and warn
        if "2023" in sql:
            print(f"⚠️ WARNING: Found hardcoded 2023 date in SQL: {sql}")

        # Detect MCC granularity level used
        mcc_granularity = "high_level"
        if "mcc_original" in sql:
            mcc_granularity = "detailed"
        if "mcc_number" in sql:
            mcc_granularity = "granular"

        return {
            "sql_query": sql,
            "query_type": "client_analysis",
            "mcc_granularity": mcc_granularity,
            "client_id": client_id,
            "original_query": user_query,
            "optimization_used": "client_id index + compound indexes + MCC intelligence",
            "date_context_applied": True
        }

    except Exception as e:
        return {"error": f"SQL generation failed: {e}"}

@tool
def generate_sql_for_benchmark_analysis(
    user_query: str,
    demographic_filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate optimized SQL for overall_transactions with current date context and enhanced MCC analysis.
    """
    ds = _ensure_datastore()
    schema = ds.get_schema_info()["overall_transactions"]

    # Get current date context
    date_context = get_current_date_context()

    # Build enhanced schema description with MCC intelligence
    schema_desc_lines = [
        "OVERALL_TRANSACTIONS TABLE:",
        f"Description: {schema['description']}",
        "",
        "KEY COLUMNS:"
    ]
    important_columns = {
        "date": "TEXT (YYYY-MM-DD) - Transaction date",
        "amount": "REAL - Transaction amount in dollars",
        "mcc_category": "TEXT - High-level spending category (Restaurants, Groceries, Transportation, etc.)",
        "mcc_original": "TEXT - Detailed merchant category (e.g., 'Eating Places and Restaurants', 'Fast Food Restaurants')",
        "mcc_number": "INTEGER - Specific MCC code (e.g., 5812 = Restaurants, 5814 = Fast Food)",
        "current_age": "INTEGER - Customer age",
        "gender": "TEXT - Customer gender",
        "yearly_income": "REAL - Annual income",
        "is_weekend": "BOOLEAN - Weekend flag",
        "is_night_txn": "BOOLEAN - Night transaction flag"
    }
    for col, desc in important_columns.items():
        schema_desc_lines.append(f"- {col}: {desc}")
    schema_desc = "\n".join(schema_desc_lines)

    # Build filter context
    filter_ctx = ""
    if demographic_filters:
        filter_lines = ["\nAPPLY THESE DEMOGRAPHIC FILTERS:"]
        for key, val in demographic_filters.items():
            if key == "gender":
                filter_lines.append(f"- gender = '{val}'")
            elif key == "age_min":
                filter_lines.append(f"- current_age >= {val}")
            elif key == "age_max":
                filter_lines.append(f"- current_age <= {val}")
            elif key == "income_min":
                filter_lines.append(f"- yearly_income >= {val}")
            elif key == "income_max":
                filter_lines.append(f"- yearly_income <= {val}")
        filter_ctx = "\n".join(filter_lines)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are an expert SQL generator for market benchmark analysis with intelligent MCC analysis.
Generate SQLite queries against overall_transactions with smart MCC granularity detection.

{date_context}

{schema_desc}
{filter_ctx}

MCC BENCHMARK ANALYSIS INTELLIGENCE:

**MCC GRANULARITY LEVELS FOR BENCHMARKS:**
1. **HIGH-LEVEL** (mcc_category): Market averages by general category
   - Examples: Average restaurant spending across demographics
   - Query: GROUP BY mcc_category

2. **DETAILED** (mcc_category + mcc_original): Specific merchant type benchmarks
   - Examples: Fast food vs fine dining market averages
   - Query: GROUP BY mcc_category, mcc_original

3. **GRANULAR** (include mcc_number): Exact MCC code market data
   - Examples: Specific MCC performance across market segments
   - Query: SELECT mcc_number, mcc_original, mcc_category

**WHEN TO USE DETAILED MCC FOR BENCHMARKS:**
- User asks for "detailed comparison", "specific merchant types vs market"
- User wants "breakdown of restaurant market", "grocery subcategory benchmarks"
- Comparative analysis: "how do I compare in fast food vs fine dining"
- Follow-up questions: "what's the market average for each restaurant type"

**BENCHMARK SQL EXAMPLES:**

High-level market averages:
SELECT mcc_category, AVG(amount) as market_avg, COUNT(*) as market_transactions
FROM overall_transactions
WHERE [demographic_filters]
GROUP BY mcc_category
ORDER BY market_avg DESC;

Detailed benchmark breakdown:
SELECT mcc_category, mcc_original, AVG(amount) as market_avg, 
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) as market_median
FROM overall_transactions
WHERE [demographic_filters]
GROUP BY mcc_category, mcc_original
ORDER BY mcc_category, market_avg DESC;

Granular MCC market analysis:
SELECT mcc_number, mcc_original, mcc_category, 
       AVG(amount) as market_avg, COUNT(*) as frequency
FROM overall_transactions
WHERE [demographic_filters]
GROUP BY mcc_number, mcc_original, mcc_category
ORDER BY market_avg DESC;

Restaurant-specific market benchmarks:
SELECT mcc_original, AVG(amount) as market_avg, COUNT(*) as market_volume
FROM overall_transactions
WHERE mcc_category = 'Restaurants' AND [demographic_filters]
GROUP BY mcc_original
ORDER BY market_avg DESC;

CRITICAL REQUIREMENTS:
- Use the CURRENT DATE CONTEXT above for date filtering
- NEVER use hardcoded years like 2023 - always use the current dates provided
- For category comparisons, use exact category names like 'Restaurants' (not 'restaurant')
- Apply demographic filters when provided
- Aggregates: AVG(), COUNT(), SUM(), PERCENTILE_CONT() for benchmarks
- Group by: mcc_category, mcc_original, mcc_number (based on granularity needed)
- Use meaningful aliases in your query
- Indexed columns: date, mcc_category, current_age, gender, amount
- ENSURE PROPER SQL SYNTAX - no missing parentheses or incomplete clauses
- RETURN ONLY THE SQL QUERY - no explanations or markdown

**MCC BENCHMARK DECISION LOGIC:**
1. Analyze user query for benchmark granularity needs
2. If detailed market comparison → include mcc_original
3. If granular market analysis → include mcc_number
4. If general market benchmarks → use mcc_category only
5. Always provide comparative market data for meaningful benchmarks

EXAMPLE CATEGORY NAMES: 'Restaurants', 'Groceries', 'Transportation', 'Financial Services', 'Entertainment'

Generate ONLY the SQL query for:
"""),
        ("human", "{user_query}")
    ])

    try:
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),            
            temperature=0,
        )  

         
        resp = llm.invoke(prompt.format_messages(user_query=user_query))
        
        if not resp or not resp.content:
            return {"error": "No response from LLM"}
            
        sql = resp.content.strip()

        # Clean up
        lines = [l.strip() for l in sql.splitlines()]
        sql_lines, found = [], False
        for line in lines:
            if line.upper().startswith("SELECT") or found:
                found = True
                sql_lines.append(line)
        sql = "\n".join(sql_lines).strip("```sql ").strip("```")

        if not sql.upper().startswith("SELECT"):
            idx = resp.content.upper().find("SELECT")
            if idx >= 0:
                sql = resp.content[idx:].split("\n\n")[0].strip()
            else:
                return {"error": f"No valid SQL query generated for: {user_query}"}

        # Validate we have a meaningful SQL query
        if not sql or len(sql.strip()) < 10:
            return {"error": f"Generated SQL too short or empty: {sql}"}

        # Check for hardcoded 2023 dates and warn
        if "2023" in sql:
            print(f"⚠️ WARNING: Found hardcoded 2023 date in SQL: {sql}")

        # Basic syntax validation
        if sql.count("(") != sql.count(")"):
            return {"error": f"SQL syntax error - mismatched parentheses: {sql}"}

        # Detect MCC granularity level used
        mcc_granularity = "high_level"
        if "mcc_original" in sql:
            mcc_granularity = "detailed"
        if "mcc_number" in sql:
            mcc_granularity = "granular"

        return {
            "sql_query": sql,
            "query_type": "benchmark_analysis",
            "mcc_granularity": mcc_granularity,
            "demographic_filters": demographic_filters,
            "original_query": user_query,
            "optimization_used": "demographic indexes + compound indexes + MCC intelligence",
            "date_context_applied": True
        }

    except Exception as e:
        return {"error": f"Benchmark SQL generation failed: {e}"}

@tool
def generate_sql_for_budget_analysis(
    user_query: str,
    client_id: int,
    analysis_type: str = "budget_performance"
) -> Dict[str, Any]:
    """
    Generate optimized SQL for budget-related queries.
    """
    ds = _ensure_datastore()
    
    # Get current date context
    date_context = get_current_date_context()
    
    # Get schema info for budget tables
    budget_schema = ds.get_schema_info().get("user_budgets", {})
    tracking_schema = ds.get_schema_info().get("budget_tracking", {})
    
    # Build comprehensive schema description
    schema_desc = f"""
BUDGET ANALYSIS TABLES:

USER_BUDGETS TABLE:
- client_id: INTEGER - Client identifier (ALWAYS required)
- category: TEXT - Budget category (matches mcc_category from transactions)
- monthly_limit: REAL - Monthly budget limit in dollars
- budget_type: TEXT - Type of budget (fixed, percentage, goal_based)
- is_active: BOOLEAN - Whether budget is currently active (use = 1)

BUDGET_TRACKING TABLE:
- client_id: INTEGER - Client identifier (ALWAYS required)
- month: TEXT - Month in YYYY-MM format
- category: TEXT - Spending category
- budgeted_amount: REAL - Budgeted amount for the month
- actual_amount: REAL - Actual spending for the month
- variance_amount: REAL - Difference (actual - budgeted)
- variance_percentage: REAL - Percentage variance

CLIENT_TRANSACTIONS TABLE (for real-time calculations):
- client_id: INTEGER - Client identifier
- date: TEXT (YYYY-MM-DD) - Transaction date
- amount: REAL - Transaction amount
- mcc_category: TEXT - Spending category
- mcc_original: TEXT - Exact Category
- yearly_income: REAL - Yearly Income 
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are an expert SQL generator for budget analysis queries.
Generate efficient SQLite queries for budget management.

{date_context}

{schema_desc}

ANALYSIS TYPES:
- "budget_performance": Compare budgeted vs actual spending
- "budget_status": Current budget limits and categories
- "budget_variance": Identify over/under spending
- "budget_trends": Historical budget performance

CRITICAL REQUIREMENTS:
- ALWAYS include WHERE client_id = {client_id}
- Use CURRENT DATE CONTEXT for date filtering
- For current month analysis, use strftime('%Y-%m', date) for month comparison
- Join tables when needed for comprehensive analysis
- Use meaningful aliases
- RETURN ONLY THE SQL QUERY - no explanations

COMMON PATTERNS:
- Current month spending: WHERE strftime('%Y-%m', date) = strftime('%Y-%m', 'now')
- Budget vs actual: JOIN user_budgets with aggregated client_transactions
- Variance calculation: (actual_amount - budgeted_amount) AS variance

Generate ONLY the SQL query for:
"""),
        ("human", "{user_query}")
    ])

    try:
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),            
            temperature=0,
        )   
        
    #     llm = ChatOpenAI(
    # model="gpt-4o-mini",   
    # temperature=0,
    # openai_api_key=os.getenv("OPENAI_API_KEY"),
    # )   
        resp = llm.invoke(prompt.format_messages(user_query=user_query))
        print(f"This is LLM {resp.content}")
        
        if not resp or not resp.content:
            return {"error": "No response from LLM"}
            
        sql = resp.content.strip()

        # Clean up SQL
        lines = [l.strip() for l in sql.splitlines()]
        sql_lines, found = [], False
        for line in lines:
            if line.upper().startswith("SELECT") or found:
                found = True
                sql_lines.append(line)
        sql = "\n".join(sql_lines).strip("```sql ").strip("```")

        # Fallback extraction
        if not sql.upper().startswith("SELECT"):
            idx = resp.content.upper().find("SELECT")
            if idx >= 0:
                sql = resp.content[idx:].split("\n\n")[0].strip()
            else:
                return {"error": f"No valid SQL query generated for: {user_query}"}

        # Validate we have a meaningful SQL query
        if not sql or len(sql.strip()) < 10:
            return {"error": f"Generated SQL too short or empty: {sql}"}

        # Validate client_id presence
        if f"client_id = {client_id}" not in sql:
            return {"error": f"Missing client_id filter in budget query: {sql}"}

        return {
            "sql_query": sql,
            "query_type": "budget_analysis",
            "analysis_type": analysis_type,
            "client_id": client_id,
            "original_query": user_query,
            "optimization_used": "budget table indexes",
            "date_context_applied": True
        }

    except Exception as e:
        return {"error": f"Budget SQL generation failed: {e}"}

@tool
def execute_generated_sql(
    sql_query: str,
    query_type: str,
    format_results: bool = True
) -> Dict[str, Any]:
    """
    Execute generated SQL query with performance monitoring and result formatting.
    """
    ds = _ensure_datastore()

    try:
        if not sql_query or not sql_query.strip():
            return {
                "error": "Empty or null SQL query provided",
                "query_executed": sql_query,
                "query_type": query_type,
                "execution_timestamp": datetime.now().isoformat()
            }
            
        print(f"[DEBUG] Executing SQL: {sql_query[:100]}...")
        start = datetime.now()
        rows, cols = ds.execute_sql_query(sql_query)
        duration = (datetime.now() - start).total_seconds()

        # Safe handling of rows and cols
        if rows is None:
            rows = []
        if cols is None:
            cols = []

        results = (
            [dict(zip(cols, r)) for r in rows]
            if format_results and rows and cols else
            rows
        )

        print(f"[DEBUG] SQL executed: {len(rows)} rows in {duration:.3f}s")
        return {
            "query_executed": sql_query,
            "query_type": query_type,
            "column_names": cols,
            "results": results,
            "row_count": len(rows),
            "execution_time_seconds": round(duration, 3),
            "execution_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"[DEBUG] SQL execution failed: {e}")
        return {
            "error": f"SQL execution failed: {e}",
            "query_executed": sql_query,
            "query_type": query_type,
            "execution_timestamp": datetime.now().isoformat()
        }


@tool
def create_or_update_budget(
    client_id: int,
    category: str,
    monthly_limit: float,
    budget_type: str = "fixed"
) -> Dict[str, Any]:
    """
    Create a new budget or update existing budget for a client and category.
    
    Args:
        client_id: The client's unique identifier
        category: Budget category (e.g., 'Groceries', 'Restaurants', 'Gas')
        monthly_limit: Monthly spending limit in dollars
        budget_type: Type of budget ('fixed', 'percentage', 'goal_based')
    
    Returns:
        Dict with success status, action taken, and details
    """
    try:
        # Get data store instance
        ds = _ensure_datastore()
        
        if not ds:
            return {
                "success": False,
                "error": "Data store not available",
                "message": "Cannot access budget data"
            }
        
        # Validate inputs
        if monthly_limit <= 0:
            return {
                "success": False,
                "error": "Invalid amount",
                "message": "Monthly limit must be greater than 0"
            }
        
        # Normalize category name
        category = category.title().strip()
        
        # Execute the create/update operation
        result = ds.create_or_update_budget(
            client_id=client_id,
            category=category,
            monthly_limit=monthly_limit,
            budget_type=budget_type
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to create/update budget for {category}"
        }
    

@tool
def delete_budget(client_id: int, category: str) -> Dict[str, Any]:
    """Delete (soft-deactivate) a budget for a specific category."""
    try:
        ds = _ensure_datastore()
        if not ds:
            return {
                "success": False,
                "error": "Data store not available",
                "message": "Cannot access budget data"
            }

        if not category or not isinstance(category, str):
            return {
                "success": False,
                "error": "Invalid category",
                "message": "Provide a non-empty category name"
            }

        # Keep behavior consistent with create/update
        category = category.title().strip()

        return ds.delete_budget(client_id=client_id, category=category)

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to delete budget for {category if category else 'unknown category'}"
        }


@tool
def list_all_budgets(
    client_id: int
) -> Dict[str, Any]:
    """
    List all budgets for a client with detailed tracking information.
    
    Args:
        client_id: The client's unique identifier
    
    Returns:
        Dict with budget list and summary information
    """
    try:
        # Get data store instance
        ds = _ensure_datastore()
        
        if not ds:
            return {
                "success": False,
                "error": "Data store not available",
                "message": "Cannot access budget data"
            }
        
        # Get detailed budget information
        result = ds.list_client_budgets_detailed(client_id)
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve budget list"
        }


@tool
def bulk_budget_operations(
    client_id: int,
    operations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Perform multiple budget operations in batch.
    
    Args:
        client_id: The client's unique identifier
        operations: List of operations, each with 'action', 'category', and optional 'monthly_limit', 'budget_type'
                   Actions: 'create', 'update', 'delete'
    
    Returns:
        Dict with results of all operations
    """
    try:
        # Get data store instance
        ds = _ensure_datastore()

        
        if not ds:
            return {
                "success": False,
                "error": "Data store not available",
                "message": "Cannot access budget data"
            }
        
        results = []
        successful_operations = 0
        
        for i, operation in enumerate(operations):
            try:
                action = operation.get('action', '').lower()
                category = operation.get('category', '').title().strip()
                
                if not category:
                    results.append({
                        "index": i,
                        "success": False,
                        "error": "Missing category",
                        "action": action
                    })
                    continue
                
                if action in ['create', 'update']:
                    monthly_limit = operation.get('monthly_limit', 0)
                    budget_type = operation.get('budget_type', 'fixed')
                    
                    if monthly_limit <= 0:
                        results.append({
                            "index": i,
                            "success": False,
                            "error": "Invalid monthly limit",
                            "category": category,
                            "action": action
                        })
                        continue
                    
                    result = ds.create_or_update_budget(
                        client_id, category, monthly_limit, budget_type
                    )
                    
                elif action == 'delete':
                    result = ds.delete_budget(client_id, category)
                    
                else:
                    results.append({
                        "index": i,
                        "success": False,
                        "error": f"Unknown action: {action}",
                        "category": category,
                        "action": action
                    })
                    continue
                
                # Add index and action to result
                result["index"] = i
                result["action"] = action
                results.append(result)
                
                if result.get("success", False):
                    successful_operations += 1
                    
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e),
                    "category": operation.get('category', 'Unknown'),
                    "action": operation.get('action', 'Unknown')
                })
        
        return {
            "success": successful_operations > 0,
            "total_operations": len(operations),
            "successful_operations": successful_operations,
            "failed_operations": len(operations) - successful_operations,
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Bulk budget operations failed",
            "results": []
        }

@tool
def update_budget_tracking_for_month(
    client_id: int,
    month: str
) -> Dict[str, Any]:
    """
    Update budget tracking calculations for a specific client and month.
    Format: month should be 'YYYY-MM' (e.g., '2025-07')
    """
    ds = _ensure_datastore()
    
    try:
        success = ds.update_budget_tracking(client_id, month)
        
        if success:
            # Get the updated tracking data
            performance_df = ds.get_budget_performance(client_id, month)
            
            return {
                "success": True,
                "message": f"Budget tracking updated for {month}",
                "client_id": client_id,
                "month": month,
                "categories_tracked": len(performance_df),
                "tracking_data": performance_df.to_dict('records') if not performance_df.empty else []
            }
        else:
            return {
                "success": False,
                "error": "Failed to update budget tracking - no active budgets found",
                "client_id": client_id,
                "month": month
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Budget tracking update failed: {e}",
            "client_id": client_id,
            "month": month
        }