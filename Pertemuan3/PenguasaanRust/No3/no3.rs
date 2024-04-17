//1. Struct dan Function untuk Menghitung Luas SegitigaStruct dan Function untuk Menghitung Luas Segitiga
struct Triangle {
    base: f64,
    height: f64,
}

impl Triangle {
    fn new(base: f64, height: f64) -> Self {
        Triangle { base, height }
    }

    fn area(&self) -> f64 {
        0.5 * self.base * self.height
    }
}

//2. Struct dan Function untuk Menghitung Volume Kubus
struct Cube {
    side: f64,
}

impl Cube {
    fn new(side: f64) -> Self {
        Cube { side }
    }

    fn volume(&self) -> f64 {
        self.side * self.side * self.side
    }
}

//3. Struct dan Function untuk Mengkonversi Celsius ke Fahrenheit
struct Temperature {
    celsius: f64,
}

impl Temperature {
    fn new(celsius: f64) -> Self {
        Temperature { celsius }
    }

    fn to_fahrenheit(&self) -> f64 {
        self.celsius * 9.0 / 5.0 + 32.0
    }
}

//4. Struct dan Function untuk Menghitung Jarak Antara Dua Titik
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    fn distance_to(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

//5. Struct dan Function untuk Mengelola Daftar Belanja
struct ShoppingList {
    items: Vec<String>,
}

impl ShoppingList {
    fn new() -> Self {
        ShoppingList { items: Vec::new() }
    }

    fn add_item(&mut self, item: &str) {
        self.items.push(item.to_string());
    }

    fn print_list(&self) {
        for item in &self.items {
            println!("{}", item);
        }
    }
}

//6. Struct dan Function untuk Mengelola Informasi Pengguna
struct User {
    username: String,
    email: String,
    age: u32,
}

impl User {
    fn new(username: &str, email: &str, age: u32) -> Self {
        User {
            username: username.to_string(),
            email: email.to_string(),
            age,
        }
    }

    fn print_info(&self) {
        println!("Username: {}", self.username);
        println!("Email: {}", self.email);
        println!("Age: {}", self.age);
    }
}

//7. Struct dan Function untuk Menghitung Gaji Karyawan
#[allow(dead_code)]
struct Employee {
    name: String,
    salary_per_hour: f64,
    hours_worked: f64,
}

impl Employee {
    fn new(name: &str, salary_per_hour: f64, hours_worked: f64) -> Self {
        Employee {
            name: name.to_string(),
            salary_per_hour,
            hours_worked,
        }
    }

    fn calculate_salary(&self) -> f64 {
        self.salary_per_hour * self.hours_worked
    }
}

//8. Struct dan Function untuk Menghitung Nilai Rata-rata
struct Statistics {
    data: Vec<f64>,
}

impl Statistics {
    fn new(data: Vec<f64>) -> Self {
        Statistics { data }
    }

    fn average(&self) -> f64 {
        let sum: f64 = self.data.iter().sum();
        sum / self.data.len() as f64
    }
}

//9. Struct dan Function untuk Menampilkan Informasi Produk
struct Product {
    name: String,
    price: f64,
}

impl Product {
    fn new(name: &str, price: f64) -> Self {
        Product {
            name: name.to_string(),
            price,
        }
    }

    fn display_info(&self) {
        println!("Produk: {}", self.name);
        println!("Harga: Rp{}", self.price);
    }
}

//10. Struct dan Function untuk Menghitung Perimeter Lingkaran
struct Circle {
    radius: f64,
}

impl Circle {
    fn new(radius: f64) -> Self {
        Circle { radius }
    }

    fn perimeter(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius
    }
}

fn main() {
    println!("Contoh pembuatan dan pemanggilan struct dan function");
    println!("");

    println!("Contoh 1");
    let triangle = Triangle::new(5.0, 8.0);
    println!("Luas segitiga: {}", triangle.area());
    println!("");

    println!("Contoh 2");
    let cube = Cube::new(3.0);
    println!("Volume kubus: {}", cube.volume());
    println!("");

    println!("Contoh 3");
    let temp = Temperature::new(25.0);
    println!(
        "Celsius: {} -> Fahrenheit: {}",
        temp.celsius,
        temp.to_fahrenheit()
    );
    println!("");

    println!("Contoh 4");
    let p1 = Point::new(1.0, 2.0);
    let p2 = Point::new(4.0, 6.0);
    println!("Jarak antara dua titik: {}", p1.distance_to(&p2));
    println!("");

    println!("Contoh 5");
    let mut list = ShoppingList::new();
    list.add_item("Apel");
    list.add_item("Beras");
    list.add_item("Sabun");
    list.print_list();
    println!("");

    println!("Contoh 6");
    let user = User::new("al novianti", "alnoviantirs@gmail.com", 30);
    user.print_info();
    println!("");

    println!("Contoh 7");
    let employee = Employee::new("Al Novianti", 20.0, 40.0);
    println!("Gaji karyawan: ${}", employee.calculate_salary());
    println!("");

    println!("Contoh 8");
    let data = vec![100.0, 80.0, 70.0, 90.0, 85.0];
    let stats = Statistics::new(data);
    println!("Nilai rata-rata: {}", stats.average());
    println!("");

    println!("Contoh 9");
    let product = Product::new("Laptop", 12000000.0);
    product.display_info();
    println!("");

    println!("Contoh 10");
    let circle = Circle::new(5.0);
    println!("Keliling lingkaran: {}", circle.perimeter());
    println!("");
}
