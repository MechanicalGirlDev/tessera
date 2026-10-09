//! MJCF actuator XML import and scalar transmission resolution.

use super::*;

pub(in crate::mjcf) fn import(
    xml: &str,
    joints: &[MjcfJointInfo],
) -> Result<MjcfActuators, MjcfLoadError> {
    let defaults = collect_defaults(xml)?;
    let mut reader = Reader::from_str(xml);
    let mut inside = false;
    let mut entries = Vec::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) if e.name().as_ref() == b"actuator" => inside = true,
            Ok(Event::End(e)) if e.name().as_ref() == b"actuator" => inside = false,
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) if inside => {
                let kind = match e.name().as_ref() {
                    b"motor" => "motor",
                    b"position" => "position",
                    b"velocity" => "velocity",
                    b"damper" => "damper",
                    b"general" => "general",
                    other => {
                        return Err(MjcfLoadError::Unsupported(format!(
                            "actuator type `{}`",
                            String::from_utf8_lossy(other)
                        )));
                    }
                };
                let attrs = resolve_defaults(
                    attributes(&e)?,
                    None,
                    DefaultElement::Actuator(kind),
                    &defaults,
                )?;
                entries.push(parse(kind, &attrs, joints, entries.len())?);
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(MjcfLoadError::Xml(e.to_string())),
            _ => {}
        }
    }
    Ok(MjcfActuators {
        controls: vec![0.0; entries.len()],
        entries,
    })
}

fn range(
    attrs: &BTreeMap<String, String>,
    prefix: &str,
) -> Result<Option<[f64; 2]>, MjcfLoadError> {
    let values = parse_optional_numbers(attrs, &format!("{prefix}range"))?;
    let range = match values.as_deref() {
        Some([lo, hi]) if lo <= hi => Some([*lo, *hi]),
        None => None,
        _ => return Err(MjcfLoadError::Invalid(format!("invalid {prefix}range"))),
    };
    match attrs.get(&format!("{prefix}limited")).map(String::as_str) {
        Some("false") => Ok(None),
        Some("true") if range.is_none() => Err(MjcfLoadError::Invalid(format!(
            "{prefix}limited needs a range"
        ))),
        Some("true" | "auto") | None => Ok(range),
        _ => Err(MjcfLoadError::Invalid(format!("invalid {prefix}limited"))),
    }
}

fn parse(
    kind: &str,
    attrs: &BTreeMap<String, String>,
    joints: &[MjcfJointInfo],
    index: usize,
) -> Result<MjcfActuatorInfo, MjcfLoadError> {
    for key in attrs.keys() {
        if !matches!(
            key.as_str(),
            "name"
                | "class"
                | "joint"
                | "gear"
                | "ctrllimited"
                | "ctrlrange"
                | "forcelimited"
                | "forcerange"
                | "kp"
                | "kv"
                | "gainprm"
                | "biasprm"
                | "gaintype"
                | "biastype"
                | "dyntype"
                | "group"
        ) {
            return Err(MjcfLoadError::Unsupported(format!(
                "actuator attribute `{key}`"
            )));
        }
    }
    if attrs.get("dyntype").is_some_and(|v| v != "none")
        || attrs.get("gaintype").is_some_and(|v| v != "fixed")
        || attrs
            .get("biastype")
            .is_some_and(|v| !matches!(v.as_str(), "none" | "affine"))
    {
        return Err(MjcfLoadError::Unsupported(
            "actuator dynamics, gain or bias type".into(),
        ));
    }
    let name = attrs
        .get("name")
        .cloned()
        .unwrap_or_else(|| format!("actuator_{index}"));
    let joint = attrs
        .get("joint")
        .ok_or_else(|| MjcfLoadError::Unsupported("non-joint actuator transmission".into()))?;
    let mut matches = joints.iter().filter(|j| &j.name == joint);
    let target = matches
        .next()
        .ok_or_else(|| MjcfLoadError::Invalid(format!("unknown actuator joint `{joint}`")))?;
    if matches.next().is_some() {
        return Err(MjcfLoadError::Invalid(format!(
            "ambiguous actuator joint `{joint}`"
        )));
    }
    if target.dofs.len() != 1 || !matches!(target.kind, JointKind::Revolute | JointKind::Prismatic)
    {
        return Err(MjcfLoadError::Unsupported(
            "actuator requires a scalar hinge or slide joint".into(),
        ));
    }
    let gear_values = parse_optional_numbers(attrs, "gear")?.unwrap_or_else(|| vec![1.0]);
    if !(gear_values.len() == 1 || gear_values.len() == 6)
        || gear_values[1..].iter().any(|v| *v != 0.0)
    {
        return Err(MjcfLoadError::Unsupported(
            "non-scalar actuator gear".into(),
        ));
    }
    let kp = optional_scalar(attrs, "kp")?.unwrap_or(1.0);
    let kv = optional_scalar(attrs, "kv")?.unwrap_or(0.0);
    if kp < 0.0 || kv < 0.0 {
        return Err(MjcfLoadError::Invalid("negative actuator gain".into()));
    }
    let gain = parse_optional_numbers(attrs, "gainprm")?
        .and_then(|v| v.first().copied())
        .unwrap_or(0.0);
    let mut bias = [0.0; 3];
    if let Some(values) = parse_optional_numbers(attrs, "biasprm")? {
        for (slot, value) in bias.iter_mut().zip(values) {
            *slot = value;
        }
    }
    let affine = attrs.get("biastype").is_some_and(|v| v == "affine") && bias[1] < 0.0;
    let dynamics = match kind {
        "motor" => MjcfActuatorKind::Motor,
        "position" => MjcfActuatorKind::Position { kp, kv },
        "velocity" => MjcfActuatorKind::Velocity { kv },
        "damper" if gain >= 0.0 => MjcfActuatorKind::Damper { gain },
        "general" if !affine || bias[2] <= 0.0 => MjcfActuatorKind::General { gain, bias, affine },
        _ => {
            return Err(MjcfLoadError::Unsupported(
                "negative actuator damping".into(),
            ));
        }
    };
    Ok(MjcfActuatorInfo {
        name,
        joint: joint.clone(),
        coordinate: target.dofs.start,
        kind: dynamics,
        gear: gear_values[0],
        control_range: range(attrs, "ctrl")?,
        force_range: range(attrs, "force")?,
    })
}
