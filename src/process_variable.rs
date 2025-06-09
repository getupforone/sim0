use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use epics_ca::{Context, ValueChannel,types::{Value,Field}};
use std::ffi::CString;
use std::collections::HashMap;

// use tokio::time::Instant;
use chrono::{DateTime, Utc};

// pub type Error = Box<dyn std::error::Error + Send + Sync>;
// pub type Result<T> = std::result::Result<T, Error>;
use anyhow::{anyhow, Result};

#[derive(Deserialize, Serialize, Debug, Clone)]
struct PVName{
    key: String,
    pvname: String,
}
#[derive(Deserialize, Serialize, Debug,Clone)]
pub struct PVNameVec {
    pv_vec: Vec<PVName>,
    
}
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PVInfos{
    pub key: String,
    pub pvname: String,
    pub wr: String,
    pub enable_ca: bool,
    pub buff_size: usize,
    pub vec_size: usize,
    pub pred_size: usize,

}
#[derive( Debug, Clone)]
pub struct PV{
    pub key: String,
    pub pvname: String,
    pub wr: u8,
    pub enable_ca: bool,
    pub buff_size: usize,
    pub vec_size: usize,
    pub pred_size: usize,
    // channel: Option<Arc<Mutex<ValueChannel<f64>>>>,
}
impl PV{
    fn new(info:PVInfos)-> PV {
        let  wr_val = match &info.wr[..] {
            "W" =>  0b10,
            "R" => 0b01,
            "WR" => 0b11,
            _ => 0b00,
        };
            
        // println!("[PV][{}] wr = {}", info.key, wr_val);
        PV {
            key: info.key,
            pvname: info.pvname,
            wr: wr_val,
            enable_ca: info.enable_ca,
            buff_size: info.buff_size,
            vec_size: info.buff_size,
            pred_size: info.pred_size,
            // channel: None,
        }
    }
}

#[derive( Debug, Clone)]


pub struct PVContext<T:Value>{
    pub key: String,
    pub pvname: String,
    pub channel:Arc<Mutex<ValueChannel<T>>>,
    pub buff_size: usize,
    pub vec_size: usize,
    pub pred_size: usize,
    pub enable_ca: bool,
}
// unsafe impl Send for PVContext {}

// unsafe impl Sync for PVContext {}

impl<T:Value+Field> PVContext<T>{
    pub async fn new(pvname:PV)-> PVContext<T> {
        let chan = Arc::new(Mutex::new(PVContext::connect(pvname.pvname.clone()).await.unwrap()));
        PVContext {
            key: pvname.key,
            pvname: pvname.pvname.clone(),
            channel: chan,
            buff_size: pvname.buff_size,
            vec_size: pvname.vec_size,
            pred_size: pvname.pred_size,
            enable_ca: pvname.enable_ca,
        }
    }

    // pub async fn connect_t(pvname:String)->Result<ValueChannel<f64>>{
    // // pub async fn connect(&mut self)->{
    //     let c_pv_string = CString::new(pvname.as_str()).unwrap();
    //     let c_str = c_pv_string.as_c_str();
    //     let ctx = Context::new().unwrap();
    //     let channel:ValueChannel<f64>  = ctx.connect::<f64>(c_str).await.unwrap();
    //     return Ok(channel)
    // }
    pub async fn connect(pvname:String)->Result<ValueChannel<T>>{
        let c_pv_string = CString::new(pvname.as_str()).unwrap();
        let c_str = c_pv_string.as_c_str();
        let ctx = Context::new().unwrap();
        let channel:ValueChannel<T>  = ctx.connect::<T>(c_str).await.unwrap();
        return Ok(channel)
    }
    pub async fn get(&mut self,channel: & Mutex<ValueChannel<T>>)->Result<T>{
        let mut value_chan = channel.lock().unwrap();//chan.unwrap();
        let value = value_chan.get().await;
        let value = match value {
            Ok(value) => value,
            Err(error) => { 
                println!("[caget] get error kind = {:?}, error severity {:?}", error.kind, error.severity);
                return  Err(anyhow::anyhow!("epics get error {}",error.into_raw()).into());
            }
            // Err(error) => panic!("Problem opening the value_channel: {:?}", error),
        };
        Ok(value)
    }

    pub async fn put(&mut self,val:T,channel: & Mutex<ValueChannel<T>>)->Result<()>{
        let mut value_chan = channel.lock().unwrap();//chan.unwrap();
        let value = value_chan.put(val).unwrap().await;

        let value = match value {
            Ok(value) => value,
            Err(error) => { 
                println!("[caget] get error kind = {:?}, error severity {:?}", error.kind, error.severity);
                return  Err(anyhow::anyhow!("epics get error {}",error.into_raw()).into());
            }
            // Err(error) => panic!("Problem opening the value_channel: {:?}", error),
        };
        Ok(value)
    }
 
}


#[derive( Debug, Clone)]
pub struct PVContextVec<T> 
    where T:Value + Field
{
    pub key: String,
    pub pvname: String,
    pub channel:Arc<Mutex<ValueChannel<[T]>>>,
}
// unsafe impl Send for PVContext {}

// unsafe impl Sync for PVContext {}

impl<T> PVContextVec<T>
where T:Value + Field
{
    pub async fn new(pvname:PV)-> PVContextVec<T> {
        let chan = Arc::new(Mutex::new(PVContextVec::connect(pvname.pvname.clone()).await.unwrap()));
        PVContextVec {
            key: pvname.key,
            pvname: pvname.pvname.clone(),
            channel: chan,
        }
    }

    // pub async fn connect_t(pvname:String)->Result<ValueChannel<f64>>{
    // // pub async fn connect(&mut self)->{
    //     let c_pv_string = CString::new(pvname.as_str()).unwrap();
    //     let c_str = c_pv_string.as_c_str();
    //     let ctx = Context::new().unwrap();
    //     let channel:ValueChannel<f64>  = ctx.connect::<f64>(c_str).await.unwrap();
    //     return Ok(channel)
    // }
    pub async fn connect(pvname:String)->Result<ValueChannel<[T]>>{
        let c_pv_string = CString::new(pvname.as_str()).unwrap();
        let c_str = c_pv_string.as_c_str();
        let ctx = Context::new().unwrap();
        let channel:ValueChannel<[T]>  = ctx.connect::<[T]>(c_str).await.unwrap();
        return Ok(channel)
    }

    pub async fn get_vec(&mut self,channel: & Mutex<ValueChannel<[T]>>)->Result<Vec<T>>{
        let mut value_chan = channel.lock().unwrap();//chan.unwrap();
        let value = value_chan.get_vec().await;
        let value = match value {
            Ok(value) => value,
            Err(error) => { 
                println!("[caget] get error kind = {:?}, error severity {:?}", error.kind, error.severity);
                return  Err(anyhow::anyhow!("epics get error {}",error.into_raw()).into());
            }
            // Err(error) => panic!("Problem opening the value_channel: {:?}", error),
        };
        Ok(value)
    }

    pub async fn put_ref(&mut self,val_ref: &Vec<T>, channel: & Mutex<ValueChannel<[T]>>)->Result<()>{
        let mut value_chan = channel.lock().unwrap();//chan.unwrap();
        let value = value_chan.put_ref(val_ref).unwrap().await;

        let value = match value {
            Ok(value) => value,
            Err(error) => { 
                println!("[caget] get error kind = {:?}, error severity {:?}", error.kind, error.severity);
                return  Err(anyhow::anyhow!("epics get error {}",error.into_raw()).into());
            }
            // Err(error) => panic!("Problem opening the value_channel: {:?}", error),
        };
        Ok(value)
    }
}



#[derive(Debug,Clone)]
pub struct PVMap{
    pub size: usize,
    pub pv_map: HashMap<String,PV>,
}
impl PVMap{
    pub fn new( pv_infos:HashMap<String,PVInfos>)->PVMap{
        let size = pv_infos.len();
        let mut pv_map:HashMap<String,PV> = HashMap::new();

        // for (key,infos) in pv_infos.iter(){

        //     let pv_inst = PV::new(infos.clone());

        //     pv_map.insert(key.clone(),pv_inst);

        // }
        // pv_infos.iter().map(|(k,v)|pv_map.insert(k.clone(),PV::new(v.clone())));
        pv_map = pv_infos.iter().map(|(k,v)|(k.clone(),PV::new(v.clone()))).collect::<HashMap<String,PV>>();
      
        PVMap{size, pv_map}
     }
}

#[derive( Debug, Clone)]
pub struct PVValue{
    pub key: String,
    pub pvname: String,
    pub value:Option<f64>,
    pub time_stamp:Option<DateTime<Utc>>,
    pub wr: u8,
    pub enable_ca: bool,
    pub buff_size: usize,
    pub vec_size: usize,
    pub pred_size: usize,
}
#[derive( Debug, Clone)]
pub struct PVVec{
    pub key: String,
    pub pvname: String,
    pub value:Option<Vec<f64>>,
    pub time_stamp:Option<DateTime<Utc>>,
    pub enable_ca: bool,
    pub buff_size: usize,
    pub vec_size: usize,
    pub pred_size: usize,
}