# TP1: Python Foundations, Clean Code, and Debugging
 
**Day:** Tuesday  
**Topic:** Coding Foundations, Clean Code Standards, and Debugging

---

## 1. Coding Foundations

* **Variable:** A name used to store data in memory so that we can reuse, modify, or delete it throughout the program.
  ```python
  age = 20      # Store in memory
  age = 21      # Modify
  del age       # Delete
  ```
* **Function:** A reusable block of code that performs a specific task.
* **Class:** A blueprint for creating objects that combines data (attributes) and behaviors (methods).
* **Module:** A single Python file (`.py`) containing variables, functions, and classes that can be imported into other files.
* **Package:** A directory containing multiple Python modules (and an `__init__.py` file) used to organize larger projects.

---

## 2. Python Clean Code (PEP 8)

Writing clean code makes programs easier to read, understand, and maintain.

### Naming Conventions
| Type | Rule | Good Example | Bad Example |
| :--- | :--- | :--- | :--- |
| **Variable** | `snake_case` | `user_name = "Sok"` | `un = "Sok"` |
| **Function** | `snake_case` | `get_total_score()` | `GetTotalScore()` |
| **Class** | `PascalCase` | `class StudentProfile:` | `class student_profile:` |
| **Constant** | `UPPER_SNAKE_CASE` | `MAX_LIMIT = 100` | `max_limit = 100` |
| **Module** | `snake_case` | `math_utils.py` | `MathUtils.py` |

### General Rules
* Use clear, descriptive names that explain what the variable or function does.
* Keep functions short and focused on a single task.
* Remove unused variables and imports.

---

## 3. Linters and Formatters

These tools check code rules and keep formatting consistent.

* **Flake8:** A linter that checks code for style mistakes, syntax errors, and PEP 8 violations.
* **Black:** A code formatter that automatically rewrites code so the spacing and line lengths follow a strict standard.
* **Ruff:** A very fast tool that acts as both a linter and a formatter (combining the jobs of Flake8 and Black).

```bash
# Check for errors
ruff check .
flake8 .

# Format code automatically
ruff format .
black .
```

---

## 4. Debugging

Debugging is the process of finding and fixing errors in code.

* **Print Statements:** Adding `print()` in the code to check variable values and track how the program runs step by step.
  ```python
  def add_numbers(a, b):
      print(f"Debug: a={a}, b={b}")
      return a + b
  ```
* **Linters (Ruff / Flake8):** Finding syntax errors, typos, or undefined variables before running the program.
* **IDE Debugger:** Using breakpoints in editors like VS Code to pause the program and inspect variables in memory line by line.