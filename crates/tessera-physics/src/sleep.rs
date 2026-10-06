//! Shared rigid-body sleep and wake thresholds.

/// Conditions under which an inactive dynamic body may be put to sleep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SleepSettings {
    /// Whether automatic sleeping is enabled.
    pub enabled: bool,
    /// Maximum linear speed considered idle, metres per second.
    pub linear_velocity_threshold: f64,
    /// Maximum angular speed considered idle, radians per second.
    pub angular_velocity_threshold: f64,
    /// Continuous idle time before sleeping, seconds.
    pub time_threshold: f64,
}

impl Default for SleepSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            linear_velocity_threshold: 0.01,
            angular_velocity_threshold: 0.01,
            time_threshold: 0.5,
        }
    }
}

impl SleepSettings {
    /// Whether every threshold is finite and physically usable.
    pub fn is_valid(self) -> bool {
        self.linear_velocity_threshold.is_finite()
            && self.linear_velocity_threshold >= 0.0
            && self.angular_velocity_threshold.is_finite()
            && self.angular_velocity_threshold >= 0.0
            && self.time_threshold.is_finite()
            && self.time_threshold > 0.0
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SleepState {
    pub(crate) sleeping: bool,
    idle_time: f64,
}

impl SleepState {
    pub(crate) fn wake(&mut self) {
        self.sleeping = false;
        self.idle_time = 0.0;
    }

    pub(crate) fn sleep(&mut self) {
        self.sleeping = true;
        self.idle_time = 0.0;
    }

    pub(crate) fn update(
        &mut self,
        settings: SleepSettings,
        dt: f64,
        linear_speed: f64,
        angular_speed: f64,
        has_contact: bool,
    ) {
        if !settings.enabled {
            self.wake();
            return;
        }
        let moving = linear_speed > settings.linear_velocity_threshold
            || angular_speed > settings.angular_velocity_threshold;
        if moving || !has_contact {
            self.wake();
            return;
        }
        if self.sleeping {
            return;
        }
        self.idle_time += dt;
        if self.idle_time >= settings.time_threshold {
            self.sleep();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_sleeps_only_after_supported_idle_time_and_wakes_on_motion() {
        let settings = SleepSettings {
            time_threshold: 0.02,
            ..Default::default()
        };
        let mut state = SleepState::default();
        state.update(settings, 0.01, 0.0, 0.0, false);
        state.update(settings, 0.01, 0.0, 0.0, true);
        assert!(!state.sleeping);
        state.update(settings, 0.01, 0.0, 0.0, true);
        assert!(state.sleeping);
        state.update(settings, 0.01, 0.1, 0.0, true);
        assert!(!state.sleeping);
    }

    #[test]
    fn disabled_sleeping_keeps_body_awake() {
        let mut state = SleepState::default();
        state.sleep();
        state.update(
            SleepSettings {
                enabled: false,
                ..Default::default()
            },
            1.0,
            0.0,
            0.0,
            true,
        );
        assert!(!state.sleeping);
    }
}
