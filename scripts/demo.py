"""
Gradio demo for Vantage text-to-SQL model
"""

import argparse
import gradio as gr
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.api import VantageAPI


# Example schemas
EXAMPLE_SCHEMAS = {
    "E-commerce": """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    created_at TIMESTAMP
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    price REAL,
    stock INTEGER
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    total REAL,
    status TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    price REAL,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
""",
    "Company HR": """
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT,
    position TEXT,
    salary REAL,
    hire_date DATE,
    manager_id INTEGER,
    FOREIGN KEY (manager_id) REFERENCES employees(id)
);

CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    budget REAL,
    location TEXT
);

CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER,
    start_date DATE,
    end_date DATE,
    budget REAL,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);
""",
    "Library": """
CREATE TABLE books (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT,
    isbn TEXT UNIQUE,
    published_year INTEGER,
    category TEXT
);

CREATE TABLE members (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    joined_date DATE,
    membership_type TEXT
);

CREATE TABLE loans (
    id INTEGER PRIMARY KEY,
    book_id INTEGER,
    member_id INTEGER,
    loan_date DATE,
    due_date DATE,
    return_date DATE,
    FOREIGN KEY (book_id) REFERENCES books(id),
    FOREIGN KEY (member_id) REFERENCES members(id)
);
""",
}

# Example questions
EXAMPLE_QUESTIONS = {
    "E-commerce": [
        "How many orders were placed last month?",
        "What are the top 5 best-selling products?",
        "List all users who have never placed an order",
        "What is the average order value by month?",
    ],
    "Company HR": [
        "How many employees are in each department?",
        "What is the average salary by department?",
        "Who are the highest paid employees?",
        "List all employees and their managers",
    ],
    "Library": [
        "How many books are currently on loan?",
        "Which members have overdue books?",
        "What are the most popular book categories?",
        "List all books published after 2020",
    ],
}


def create_demo(model_path: str, share: bool = False):
    """Create Gradio demo interface"""
    
    # Load model
    print(f"Loading model from {model_path}...")
    api = VantageAPI.from_pretrained(model_path)
    print("Model loaded successfully!")
    
    def generate_sql(question: str, schema: str) -> str:
        """Generate SQL from question and schema"""
        try:
            if not question.strip():
                return "Please enter a question."
            
            if not schema.strip():
                return "Please enter a database schema."
            
            sql = api.generate(question, schema)
            return sql
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def load_example_schema(schema_name: str) -> str:
        """Load example schema"""
        return EXAMPLE_SCHEMAS.get(schema_name, "")
    
    def load_example_question(schema_name: str) -> list:
        """Load example questions"""
        questions = EXAMPLE_QUESTIONS.get(schema_name, [])
        return gr.update(choices=questions, value=questions[0] if questions else "")
    
    # Create interface
    with gr.Blocks(title="Vantage Text-to-SQL") as demo:
        gr.Markdown("# 🎯 Vantage: Text-to-SQL with Mixture of Experts")
        gr.Markdown(
            "Convert natural language questions into SQL queries. "
            "Select an example schema or enter your own!"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                schema_dropdown = gr.Dropdown(
                    choices=list(EXAMPLE_SCHEMAS.keys()),
                    label="Example Schemas",
                    value="E-commerce",
                )
                
                schema_input = gr.Textbox(
                    label="Database Schema (CREATE TABLE statements)",
                    placeholder="CREATE TABLE users (id INT, name TEXT);",
                    lines=15,
                    value=EXAMPLE_SCHEMAS["E-commerce"],
                )
                
                question_dropdown = gr.Dropdown(
                    choices=EXAMPLE_QUESTIONS["E-commerce"],
                    label="Example Questions",
                    value=EXAMPLE_QUESTIONS["E-commerce"][0],
                )
                
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="How many users are in the database?",
                    lines=3,
                    value=EXAMPLE_QUESTIONS["E-commerce"][0],
                )
                
                generate_btn = gr.Button("🔮 Generate SQL", variant="primary")
            
            with gr.Column(scale=1):
                sql_output = gr.Code(
                    label="Generated SQL",
                    language="sql",
                    lines=20,
                )
                
                gr.Markdown("### Tips:")
                gr.Markdown(
                    "- Be specific in your questions\n"
                    "- Provide complete schema with foreign keys\n"
                    "- Always validate generated SQL before execution"
                )
        
        # Event handlers
        schema_dropdown.change(
            fn=load_example_schema,
            inputs=[schema_dropdown],
            outputs=[schema_input],
        )
        
        schema_dropdown.change(
            fn=load_example_question,
            inputs=[schema_dropdown],
            outputs=[question_dropdown],
        )
        
        question_dropdown.change(
            fn=lambda x: x,
            inputs=[question_dropdown],
            outputs=[question_input],
        )
        
        generate_btn.click(
            fn=generate_sql,
            inputs=[question_input, schema_input],
            outputs=[sql_output],
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["How many users?", "CREATE TABLE users (id INT, name TEXT);"],
                ["Average order value", "CREATE TABLE orders (id INT, total REAL);"],
            ],
            inputs=[question_input, schema_input],
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="Run Vantage demo")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run demo on"
    )
    
    args = parser.parse_args()
    
    # Create and launch demo
    demo = create_demo(args.model_path, args.share)
    
    print("\n" + "="*50)
    print("LAUNCHING DEMO")
    print("="*50)
    print(f"\nModel: {args.model_path}")
    print(f"Port: {args.port}")
    if args.share:
        print("Share: Enabled (public link will be generated)")
    print("\nStarting server...")
    
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
