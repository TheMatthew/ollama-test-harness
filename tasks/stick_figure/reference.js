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
