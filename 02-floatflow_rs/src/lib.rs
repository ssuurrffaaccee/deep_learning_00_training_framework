pub mod error;
pub mod executor;
pub mod float;
pub mod graph;
pub mod op;
pub mod ops;
pub mod optimizer;
pub mod polynomial;
pub mod store;
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
