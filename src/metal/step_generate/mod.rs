//! M2/M4: end-to-end monolithic generate loop (prefill → denoise blocks → KV extend).

mod config;
mod progress;
mod session;
mod turn;

pub use config::{CancelToken, StepGenerateConfig, StepObserver, StepProgressEvent};
pub use session::{KvSnapshot, StepGenerateSession};
pub use turn::{
    BlockOutcome, ProposedBlock, TurnState, begin_turn, commit_block, finish_turn,
    generate_monolithic, generate_with_session, propose_block,
};

#[cfg(test)]
mod tests;
