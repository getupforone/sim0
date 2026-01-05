use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::{BufReader, BufWriter}};
use std::path::Path;
use anyhow::{anyhow, Result};


#[derive(Encode, Decode, Serialize, Deserialize, Debug)]    
pub struct binData{
    pub data: HashMap<String, Vec<Vec<f32>>>,
    pub len: HashMap<String, usize>,
    pub shotnums: Vec<String>,
}

pub fn save_bin(datahash: &HashMap<String, Vec<Vec<f32>>>, lenhash: &HashMap<String, usize>,shotnums: &Vec<String>, file_path: &str) -> Result<()> {
    let wrapper = binData{data: datahash.clone(),len: lenhash.clone(),shotnums: shotnums.clone()};
    let file = File::create(file_path)?;
    let mut writer = BufWriter::new(file);  
    bincode::encode_into_std_write(wrapper, &mut writer, bincode::config::standard())?;
    Ok(())
}

pub fn load_bin(file_path: &str) -> Result<(HashMap<String, Vec<Vec<f32>>>,HashMap<String, usize>, Vec<String>)> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
    let data: binData = bincode::decode_from_std_read(&mut reader, bincode::config::standard())?;
    Ok((data.data, data.len, data.shotnums))
}

pub fn get_bin_dir(mode:&str) -> Result<String>{
    let path_str = format!("./bincode/{}",mode);
    let dir_path = Path::new(&path_str);
    if !dir_path.exists() {
        println!("Creating directory: {}", dir_path.display());
        std::fs::create_dir_all(&dir_path)?;
    } else if dir_path.is_dir(){
        println!("Directory already exists: {}", dir_path.display());
    } else {
        println!("Path exists but is not a directory: {}", dir_path.display());
        return Err(anyhow::anyhow!("Path exists but is not a directory"));
    }
    Ok(path_str)
}
pub fn get_bin_file_path(mode:&str) -> Result<String>{
    let dir_path = get_bin_dir(mode)?;
    let path_str = format!("{}/{}_data.bin", dir_path, mode);
    let file_path = Path::new(&path_str);
    if !file_path.exists() {
        println!("Creating file: {}", file_path.display());
        
    } else if file_path.is_file(){
        println!("File already exists: {}", file_path.display());
    } else {
        println!("Path exists but is not a file: {}", file_path.display());
        return Err(anyhow::anyhow!("Path exists but is not a file"));
    }
    Ok(path_str)
}

pub fn get_binfile_paths_in_folder(folder_path: &str) -> Result<Vec<String>> {
    let mut file_vec = Vec::new();
    let entries = std::fs::read_dir(folder_path)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension == "bin" {
                    if let Some(path_str) = path.to_str() {
                        file_vec.push(path_str.to_string());
                    }
                }
            }
        }
    }

    Ok(file_vec)
}