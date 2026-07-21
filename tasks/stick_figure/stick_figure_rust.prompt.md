Port the following JavaScript program to Rust. The Rust program must produce **exactly** the same output (same characters, same lines, same spacing).

```javascript
function stickFigure() {
    const width = 21;
    const mid = Math.floor(width / 2);
    let lines = [];

    // Hat
    lines.push(" ".repeat(mid - 2) + "_____");

    // Head
    lines.push(" ".repeat(mid - 2) + "/     \\");
    lines.push(" ".repeat(mid - 2) + "| o o |");
    lines.push(" ".repeat(mid - 2) + "|  >  |");
    lines.push(" ".repeat(mid - 2) + "| \\_/ |");
    lines.push(" ".repeat(mid - 2) + "\\_____/");

    // Neck
    lines.push(" ".repeat(mid) + "|");

    // Arms and torso
    lines.push(" ".repeat(mid - 5) + "-----+-----");
    lines.push(" ".repeat(mid - 1) + "/|\\");
    lines.push(" ".repeat(mid) + "|");
    lines.push(" ".repeat(mid - 1) + "/|\\");
    lines.push(" ".repeat(mid) + "|");

    // Waist
    lines.push(" ".repeat(mid - 2) + "___|___");

    // Legs
    lines.push(" ".repeat(mid - 1) + "/ \\");
    lines.push(" ".repeat(mid - 2) + "/   \\");
    lines.push(" ".repeat(mid - 3) + "/     \\");
    lines.push(" ".repeat(mid - 3) + "|     |");
    lines.push(" ".repeat(mid - 2) + "=   =");

    for (const line of lines) {
        console.log(line);
    }
}

stickFigure();
```

Requirements:
* The Rust program must be a single self-contained `main.rs` file with no external dependencies (no Cargo.toml needed, just `rustc` compilation).
* It must print the exact same output as the JavaScript version — every space, character, and line must match precisely.
* Use `println!` for output.
* Keep the code clean and idiomatic Rust.

Return only the Rust source code inside a single ```rust ... ``` block.
