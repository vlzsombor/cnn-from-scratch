use csv::{ReaderBuilder, StringRecord};
use ndarray::Array2;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug)]
pub struct ImagePixel {
    pub target: u32,
    pub pixels: Array2<f32>,
}

pub fn process_csv<P: AsRef<Path>>(file_path: P) -> Result<Vec<ImagePixel>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let buf_reader = BufReader::new(file);

    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(buf_reader);

    let mut rows = Vec::new();
    let mut record = StringRecord::new();

    while rdr.read_record(&mut record)? {
        let target = record[0].parse::<u32>()?;

        let pixels_vec: Result<Vec<f32>, _> = record.iter()
            .skip(1)
            .map(|p| p.parse::<f32>())
            .collect();

        // 784 Elemente in eine 28x28 Matrix konvertieren
        let pixels = Array2::from_shape_vec((28, 28), pixels_vec?)?;

        rows.push(ImagePixel { target, pixels });
    }

    Ok(rows)
}


pub fn normalize_image_pixels_vec(image_pixels: &[ImagePixel], max_pixel_value: u32) -> Vec<ImagePixel> {
    image_pixels.iter().map(|x| normalize_image_pixels(x, max_pixel_value)).collect()
}

pub fn normalize_image_pixels(image_pixel: &ImagePixel, max_pixel_value: u32) -> ImagePixel {
    let max_val = max_pixel_value as f32;
    ImagePixel {
        target: image_pixel.target,
        pixels: image_pixel.pixels.mapv(|x| x / max_val),
    }
}