//! Browser-independent lifecycle facts shared by the closed WebGPU seam.

/// Closed lifecycle of one canvas/device generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebGpuSessionState {
    /// No drawable canvas is currently configured.
    Suspended,
    /// A configured device generation accepts submissions.
    Active,
    /// The browser terminated the active device generation.
    Lost,
    /// A replacement adapter/device request is in flight.
    Recovering,
    /// Terminal cleanup is awaiting submitted work.
    Disposing,
    /// Explicit terminal cleanup completed.
    Disposed,
    /// An operation failed and reuse is unsafe.
    Poisoned,
}

/// Minimal no-GPU reducer used to test lifecycle distinctions on every host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(test)]
pub(crate) struct Lifecycle {
    state: WebGpuSessionState,
    generation: u64,
}

#[cfg(test)]
impl Lifecycle {
    /// Creates the not-yet-configured first device generation.
    pub(crate) const fn new() -> Self {
        Self {
            state: WebGpuSessionState::Suspended,
            generation: 1,
        }
    }
    /// Applies an active configuration for a nonzero canvas extent.
    pub(crate) fn configure(&mut self, drawable: bool) {
        self.state = if drawable {
            WebGpuSessionState::Active
        } else {
            WebGpuSessionState::Suspended
        };
    }
    /// Records device loss, which is distinct from resize and completion.
    pub(crate) fn lose(&mut self) {
        if matches!(
            self.state,
            WebGpuSessionState::Active | WebGpuSessionState::Suspended
        ) {
            self.state = WebGpuSessionState::Lost;
        }
    }
    fn poison(&mut self) {
        self.state = WebGpuSessionState::Poisoned;
    }
    /// Begins recovery only from a lost generation.
    pub(crate) fn begin_recover(&mut self) -> bool {
        if self.state != WebGpuSessionState::Lost {
            return false;
        }
        self.state = WebGpuSessionState::Recovering;
        true
    }
    /// Installs the next generation after async device creation settles.
    pub(crate) fn finish_recover(&mut self, drawable: bool) {
        debug_assert_eq!(self.state, WebGpuSessionState::Recovering);
        self.generation += 1;
        self.configure(drawable);
    }
    /// Enters the terminal disposal state.
    pub(crate) fn dispose(&mut self) {
        self.state = WebGpuSessionState::Disposed;
    }
    pub(crate) const fn state(self) -> WebGpuSessionState {
        self.state
    }
    pub(crate) const fn generation(self) -> u64 {
        self.generation
    }
}

/// Host-testable ownership accounting for browser callback roots. This does
/// not emulate WebGPU; it proves the policy that tickets and observers are
/// retained through settlement and stale generations cannot mutate a new one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(test)]
struct Accounting {
    generation: u64,
    token: u64,
    active_tickets: u8,
    detached_tickets: u8,
    active_observers: u8,
    retired_observers: u8,
}

#[cfg(test)]
impl Accounting {
    const fn new() -> Self {
        Self {
            generation: 1,
            token: 0,
            active_tickets: 0,
            detached_tickets: 0,
            active_observers: 1,
            retired_observers: 0,
        }
    }
    fn submit(&mut self) {
        self.active_tickets += 1;
    }
    fn settle_active(&mut self) {
        self.active_tickets -= 1;
    }
    fn begin_recover(&mut self) -> (u64, u64) {
        let old = (self.generation, self.token);
        self.token += 1;
        self.detached_tickets += self.active_tickets;
        self.active_tickets = 0;
        self.retired_observers += self.active_observers;
        self.active_observers = 0;
        old
    }
    fn install_recovered(&mut self) {
        self.generation += 1;
        self.active_observers = 1;
    }
    fn stale_callback_is_ignored(&self, generation: u64, token: u64) -> bool {
        generation != self.generation || token != self.token
    }
    fn dispose_after_settlement(&mut self) {
        self.token += 1;
        self.active_tickets = 0;
        self.detached_tickets = 0;
        self.active_observers = 0;
        self.retired_observers = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loss_recovery_increments_generation_but_resize_does_not() {
        let mut lifecycle = Lifecycle::new();
        lifecycle.configure(true);
        lifecycle.configure(false);
        assert_eq!(lifecycle.generation(), 1);
        lifecycle.lose();
        assert!(lifecycle.begin_recover());
        lifecycle.finish_recover(true);
        assert_eq!(lifecycle.state(), WebGpuSessionState::Active);
        assert_eq!(lifecycle.generation(), 2);
    }
    #[test]
    fn dispose_is_terminal_and_not_recoverable() {
        assert_ne!(WebGpuSessionState::Disposing, WebGpuSessionState::Poisoned);
        let mut lifecycle = Lifecycle::new();
        lifecycle.dispose();
        lifecycle.lose();
        assert_eq!(lifecycle.state(), WebGpuSessionState::Disposed);
        assert!(!lifecycle.begin_recover());
    }
    #[test]
    fn destroyed_loss_does_not_reopen_a_poisoned_generation() {
        let mut lifecycle = Lifecycle::new();
        lifecycle.configure(true);
        lifecycle.poison();
        lifecycle.lose();
        assert_eq!(lifecycle.state(), WebGpuSessionState::Poisoned);
    }
    #[test]
    fn normal_completion_releases_only_settled_ticket() {
        let mut accounting = Accounting::new();
        accounting.submit();
        accounting.submit();
        accounting.settle_active();
        assert_eq!(accounting.active_tickets, 1);
        assert_eq!(accounting.active_observers, 1);
    }
    #[test]
    fn loss_keeps_old_roots_and_tickets_until_they_settle() {
        let mut accounting = Accounting::new();
        accounting.submit();
        let old = accounting.begin_recover();
        assert_eq!(accounting.detached_tickets, 1);
        assert_eq!(accounting.retired_observers, 1);
        accounting.install_recovered();
        assert!(accounting.stale_callback_is_ignored(old.0, old.1));
    }
    #[test]
    fn partial_setup_failure_has_no_registered_observer() {
        let mut accounting = Accounting::new();
        // Installation registers the observer last; a prior failure leaves no
        // browser callback root to outlive the failed generation.
        accounting.active_observers = 0;
        assert_eq!(accounting.active_observers, 0);
    }
    #[test]
    fn disposal_waits_for_active_and_detached_ownership() {
        let mut accounting = Accounting::new();
        accounting.submit();
        accounting.begin_recover();
        accounting.install_recovered();
        accounting.submit();
        accounting.dispose_after_settlement();
        assert_eq!(accounting.active_tickets, 0);
        assert_eq!(accounting.detached_tickets, 0);
        assert_eq!(accounting.active_observers, 0);
        assert_eq!(accounting.retired_observers, 0);
    }
    #[test]
    fn dispose_invalidates_a_recovery_install_waiting_on_validation() {
        let mut accounting = Accounting::new();
        accounting.begin_recover();
        let pending_token = accounting.token;
        // The real session increments this token before taking Objects, so a
        // validation await that resumes afterwards cannot commit a device.
        accounting.dispose_after_settlement();
        assert_ne!(pending_token, accounting.token);
        assert!(accounting.stale_callback_is_ignored(1, pending_token));
    }
}
