#!/usr/bin/env python3
import os
import sys
import json
import argparse

DEFAULTS = {
    "c": {
        "extension": "c",
        "build": ["gcc", "-Wall", "-Wextra", "{src}", "-o", "{bin}"],
        "run": ["{bin}"]
    },
    "cpp": {
        "extension": "cpp",
        "build": ["g++", "-Wall", "-Wextra", "-std=c++17", "{src}", "-o", "{bin}"],
        "run": ["{bin}"]
    },
    "python": {
        "extension": "py",
        "build": None,
        "run": ["python3", "{src}"]
    },
    "go": {
        "extension": "go",
        "build": ["go", "build", "-o", "{bin}", "{src}"],
        "run": ["{bin}"]
    },
    "rust": {
        "extension": "rs",
        "build": ["rustc", "{src}", "-o", "{bin}"],
        "run": ["{bin}"]
    },
    "javascript": {
        "extension": "js",
        "build": None,
        "run": ["node", "{src}"]
    }
}

def main():
    parser = argparse.ArgumentParser(description="Generate a new test task for ollama-test-harness")
    parser.add_argument("--name", help="Name of the task (e.g. fibonacci)")
    parser.add_argument("--language", help="Programming language (e.g. c, python, go)")
    parser.add_argument("--harness-path", default="/home/matthew/work/git/ollama-test-harness", help="Path to the ollama-test-harness directory")
    
    args = parser.parse_args()
    
    name = args.name
    language = args.language
    
    if not name:
        name = input("Enter task name (e.g., fibonacci): ").strip().lower()
        if not name:
            print("Error: Task name is required.")
            sys.exit(1)
            
    if not language:
        language = input("Enter programming language (e.g., c, python, go, rust): ").strip().lower()
        if not language:
            print("Error: Language is required.")
            sys.exit(1)

    # Clean name
    name = name.replace(" ", "_")
    
    lang_defaults = DEFAULTS.get(language, {
        "extension": language,
        "build": None,
        "run": [language, "{src}"]
    })
    
    task_dir = os.path.join(args.harness_path, "tasks", name)
    os.makedirs(task_dir, exist_ok=True)
    
    prompt_filename = f"{name}_{language}.prompt.md"
    json_filename = f"{name}_{language}.json"
    
    prompt_path = os.path.join(task_dir, prompt_filename)
    json_path = os.path.join(task_dir, json_filename)
    
    # Write empty / template prompt markdown file
    prompt_content = f"""Write a simple, self-contained {language.upper()} program that implements the {name.replace('_', ' ').title()} problem.

The program should meet the following requirements:
* [Requirement 1]
* [Requirement 2]

Provide the source code wrapped cleanly inside a single ```{language} ... ``` block.
"""
    
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_content)
        
    # Write JSON config
    config = {
        "name": f"{name}_{language}",
        "language": language,
        "file_extension": lang_defaults["extension"],
        "prompt_file": prompt_filename,
        "system_prompt": "You are a careful {language} programmer. Respond only with the requested code, wrapped in a single ```{language} ... ``` block, and no explanation outside the code block.",
        "build_command": lang_defaults["build"],
        "run_command": lang_defaults["run"],
        "validation": [
            {
                "type": "line_count",
                "expected": 10
            }
        ]
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    print(f"Successfully generated new task scaffold in:")
    print(f"  Folder:  {task_dir}")
    print(f"  Config:  {json_path}")
    print(f"  Prompt:  {prompt_path}")
    print("\nNext steps:")
    print(f"1. Update the requirements in {prompt_filename}")
    print(f"2. Add validation rules (line_count, line_width, single_char_at, symmetric_pair) in {json_filename}")
    print(f"3. Run the benchmark: python3 ollama-test.py --task tasks/{name}/{json_filename}")

if __name__ == "__main__":
    main()
