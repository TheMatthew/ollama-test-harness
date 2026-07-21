fn main() {
    let width = 21;
    let mid = width / 2;

    // Hat
    println!("{}_____", " ".repeat(mid - 2));

    // Head
    println!("{}/     \\", " ".repeat(mid - 2));
    println!("{}| o o |", " ".repeat(mid - 2));
    println!("{}|  >  |", " ".repeat(mid - 2));
    println!("{}| \\_/ |", " ".repeat(mid - 2));
    println!("{}\\_____/", " ".repeat(mid - 2));

    // Neck
    println!("{}|", " ".repeat(mid));

    // Arms and torso
    println!("{}-----+-----", " ".repeat(mid - 5));
    println!("{}/|\\", " ".repeat(mid - 1));
    println!("{}|", " ".repeat(mid));
    println!("{}/|\\", " ".repeat(mid - 1));
    println!("{}|", " ".repeat(mid));

    // Waist
    println!("{}___|___", " ".repeat(mid - 2));

    // Legs
    println!("{}/ \\", " ".repeat(mid - 1));
    println!("{}/   \\", " ".repeat(mid - 2));
    println!("{}/     \\", " ".repeat(mid - 3));
    println!("{}|     |", " ".repeat(mid - 3));
    println!("{}=   =", " ".repeat(mid - 2));
}
