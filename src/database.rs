
use std::collections::{HashMap,VecDeque};
use std::sync::{Arc, Mutex};
// use tokio::time::{self, Duration,Instant};
use tracing::{debug, error, info, instrument};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub struct KVT{
    pub key: String,
    pub value: f64,
    pub time_stamp: DateTime<Utc>,
}

#[derive(Debug)]
pub struct DbDropGuard{
    db: Db,
}
#[derive(Debug, Clone)]
pub struct Db {
    shared:Arc<Shared>,
}
#[derive(Debug)]
struct Shared{
    state:Mutex<State>,
}
#[derive(Debug)]
struct State{
    entries: HashMap<String, Entry>,
    shutdown: bool,
}
#[derive(Debug, Clone)]
pub struct Item{
    pub value: f64,
    pub time_stamp: DateTime<Utc>,
}
#[derive(Debug)]
pub struct Entry{
    buffer: RingBuff<Item>,
}
#[derive(Debug,Clone)]
pub struct RingBuff<T: Clone>{
    pub buff: VecDeque<T>,
    size: usize,
}
impl<T: Clone> RingBuff<T>{
    pub fn new(size:usize) -> RingBuff<T>{
        RingBuff{
            buff : VecDeque::with_capacity(size),
            size : size,
        }
    }
    pub fn push_back(&mut self, value:T)->Option<T>{

        self.buff.push_back(value);
        while self.buff.len() > self.size {
            return self.buff.pop_front();
        }
        return None
    }
    pub fn pop_front(&mut self)-> Option<T>{
        self.buff.pop_front()
    }
    pub fn front(& self)-> Option<T>{
        match self.buff.front(){
            Some(rvalue) =>Some(rvalue.clone()),
            None =>None,
        }
    }
    pub fn len(&self)-> usize{
        self.buff.len()
    }
    pub fn range(&self, sz:usize)-> Option<Vec<T>>{
        if self.buff.len() >= sz {
            // let range: Vec<T> = self.buff.iter().rev().take(sz).cloned().collect();
            let range: Vec<T> = self.buff.iter().take(sz).cloned().collect();
            Some(range)
        } else {
            None
        }
    }
    pub fn rev_range(& self, sz:usize)-> Option<Vec<T>>{
        if self.buff.len() >= sz {
            let range: Vec<T> = self.buff.iter().rev().take(sz).rev().cloned().collect();
            // let range: Vec<T> = self.buff.iter().take(sz).cloned().collect();
            Some(range)
        } else {
            None
        }
    }
}
impl DbDropGuard{
    pub fn new() -> DbDropGuard{
        DbDropGuard{db: Db::new()}
    }
    pub fn db(&self) -> Db{
        self.db.clone()
    }
}

impl Db{
    pub fn new() -> Db{
        let shared = Arc::new(Shared {
            state: Mutex::new( State{
                entries: HashMap::new(),
                shutdown: false,
            }),
        });

        Db{ shared }
    }
    pub fn get_buff(&self, key: &str) -> Option<RingBuff<Item>>{
        let state = self.shared.state.lock().unwrap();
        // let buff = state.entries.get(key).unwrap().data.clone();
        let mut r_val = None;
        if !state.entries.contains_key(key) {
            error!("Db does not contains key {}", key);
            r_val = None;
        }else{
            r_val= state.entries.get(key).map(|entry|entry.buffer.clone());
        }
        drop(state);
        r_val
    }
    pub fn set(&self, key: &String, value:RingBuff<Item>){
        let mut state = self.shared.state.lock().unwrap();

        let prev = state.entries.insert(
            key.clone(),
            Entry{buffer:value},
        );
        debug!("[Db][set] key = {}", key);
        if let Some(prev) = prev{
            info!("[Db][set] key set twics previous entry modified [{}]", key);
        }
        drop(state);

    }
    pub fn contained(&self, key: &String)-> bool{
        let state = self.shared.state.lock().unwrap();
        let mut r_val = false;
        if state.entries.contains_key(key){
            r_val = true;
        }else{
            r_val = false;
        }
        drop(state);
        r_val
    }
    pub fn len(&self, key:&str) -> Option<usize>{
        let mut state = self.shared.state.lock().unwrap();
        let len = state.entries.get(key).map(|entry|entry.buffer.len());

        drop(state);
        len
    }
    pub fn pop(&self, key:&str) -> Option<Item>{
        let mut state = self.shared.state.lock().unwrap();
        let mut value = state.entries.get_mut(key).map(|entry|entry.buffer.pop_front());
        let r_val = match value{
            Some(val) =>{val},
            None =>{
                // println!("[Db][pop][key :{}]pop failed, does not have item in buff", key);
                None
            }
        };
        drop(state);
        r_val
    }
    pub fn get_front(&self, key:&str) -> Option<Item>{
        let mut state = self.shared.state.lock().unwrap();
        let value = state.entries.get(key).map(|entry|entry.buffer.front());
        let r_val = match value{
            Some(val) =>{println!("[get_front][{}] value: {}",key, val.clone().unwrap().value);
            val
            },
            None =>None,
        };
        drop(state);
        r_val
    }
    pub fn get_vec(&self, key:&str,size:usize) -> Option<Vec<Item>>{
        let mut state = self.shared.state.lock().unwrap();
        let r_val = state.entries.get(key)
                                        .map(|entry|entry.buffer.range(size))?;

        drop(state);
        r_val
    }
    pub fn get_vec_end(&self, key:&str,size:usize) -> Option<Vec<Item>>{
        let mut state = self.shared.state.lock().unwrap();
        let r_val = state.entries.get(key)
                                        .map(|entry|entry.buffer.rev_range(size))?;

        drop(state);
        r_val
    }
    pub fn push(&self, key:&str, value: Item){
        let mut state = self.shared.state.lock().unwrap();
        state.entries.get_mut(key).map(|entry|{
            // println!("[Db][push][{}] = state.entriess {},",key,value.value );
            if let Some(pop_value) = entry.buffer.push_back(value){
                debug!("[Db][push][key :{}]buffer is full so {:?} is pop out", key, pop_value);
            }
        });
        // println!("[Db][push] = state.entriess {:?}",state.entries );

        drop(state);

    }

}