mod helper; // Import module helper.rs

use helper::multiply; // Import fungsi multiply dari helper.rs

fn main() {
    let result = multiply(4, 5);
    println!("Hasil perkalian: {}", result);
}