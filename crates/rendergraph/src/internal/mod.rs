//! Private render-graph data model and callback retention boundary.
//!
//! This module carries declaration data between authoring and compilation; it never exposes
//! native backend objects. Retained callbacks own setup data so compiled plans remain valid
//! after graph construction returns.

#![allow(dead_code)]

mod declarations;
mod model;
mod retained;

pub(crate) use declarations::*;
pub(crate) use model::*;
pub(crate) use retained::*;
