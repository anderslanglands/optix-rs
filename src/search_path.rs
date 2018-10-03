use config::Config;
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

pub struct SearchPath {
    paths: Vec<PathBuf>,
}

#[derive(Debug)]
pub enum Error {
    FileNotFound(String),
}

impl fmt::Display for Error {
    fn fmt(&self, output: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::FileNotFound(file) => {
                write!(output, "File {} could not be found", file)
            }
        }
    }
}

impl SearchPath {
    pub fn new() -> SearchPath {
        SearchPath { paths: Vec::new() }
    }

    pub fn from_config_env(var_name: &str) -> SearchPath {
        let mut conf = Config::default();
        conf.merge(config::Environment::new()).ok();
        let conf_map = conf
            .try_into::<HashMap<String, String>>()
            .unwrap_or(HashMap::new());
        let paths = conf_map
            .get(var_name)
            .unwrap()
            .split(";")
            .map(|s| PathBuf::from(s))
            .collect();
        SearchPath { paths }
    }

    pub fn from_config_file(file_name: &str, var_name: &str) -> SearchPath {
        let mut conf = Config::default();
        conf.merge(config::File::with_name(file_name).required(false))
            .unwrap();
        conf.merge(config::Environment::new()).ok();
        let conf_map = conf
            .try_into::<HashMap<String, String>>()
            .unwrap_or(HashMap::new());
        let paths = conf_map
            .get(var_name)
            .unwrap()
            .split(";")
            .map(|s| PathBuf::from(s))
            .collect();
        SearchPath { paths }
    }

    pub fn add_path(&mut self, path: PathBuf) {
        self.paths.push(path);
    }

    pub fn exists(&self, filename: &str) -> bool {
        self.paths.iter().any(|p| {
            let mut pb = p.clone();
            pb.push(filename);
            pb.exists()
        })
    }

    pub fn open(&self, filename: &str) -> Result<std::fs::File, Error> {
        for p in &self.paths {
            let mut pb = p.clone();
            pb.push(filename);
            println!("Checking {}", pb.to_str().unwrap());
            match std::fs::File::open(pb) {
                Ok(f) => return Ok(f),
                Err(err) => {
                    println!("Error: {}", err);
                },
            }
        }
        Err(Error::FileNotFound(filename.to_owned()))
    }

    pub fn paths(&self) -> &Vec<PathBuf> {
        &self.paths
    }
}
