//! MJCF actuator XML import and scalar transmission resolution.

use super::*;

pub(in crate::mjcf) fn import(
    xml: &str,
    joints: &[MjcfJointInfo],
) -> Result<MjcfActuators, MjcfLoadError> {
    let mut defaults = collect_defaults(xml)?;
    // The body importer only collects native actuator shortcuts. Keep other
    // shortcut defaults here as metadata without changing body import policy.
    let mut reader = Reader::from_str(xml);
    let mut classes = Vec::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) if e.name().as_ref() == b"default" => {
                classes.push(attributes(&e)?.get("class").cloned().unwrap_or_default());
            }
            Ok(Event::End(e)) if e.name().as_ref() == b"default" => {
                let _closed = classes.pop();
            }
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) if !classes.is_empty() => {
                let kind = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                if !matches!(
                    kind.as_str(),
                    "default" | "motor" | "position" | "velocity" | "damper" | "general"
                ) {
                    let class = defaults
                        .get_mut(classes.last().map(String::as_str).unwrap_or_default())
                        .ok_or_else(|| {
                            MjcfLoadError::Invalid("unknown actuator default class".into())
                        })?;
                    class
                        .actuators
                        .entry(kind)
                        .or_default()
                        .extend(attributes(&e)?);
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(MjcfLoadError::Xml(e.to_string())),
            _ => {}
        }
    }
    let mut reader = Reader::from_str(xml);
    let mut inside = false;
    let mut nested = 0;
    let mut entries = Vec::new();
    loop {
        let event = reader.read_event();
        let starts = matches!(&event, Ok(Event::Start(_)));
        match event {
            Ok(Event::Start(e)) if e.name().as_ref() == b"actuator" => inside = true,
            Ok(Event::End(e)) if e.name().as_ref() == b"actuator" => inside = false,
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) if inside && nested == 0 => {
                let kind = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                let explicit = attributes(&e)?;
                let selected = explicit
                    .get("class")
                    .map(String::as_str)
                    .unwrap_or_default();
                let mut class = defaults.get(selected).ok_or_else(|| {
                    MjcfLoadError::Invalid(format!("unknown default class `{selected}`"))
                })?;
                let mut chain = vec![class];
                while let Some(parent) = &class.actuator_parent {
                    class = defaults.get(parent).ok_or_else(|| {
                        MjcfLoadError::Invalid(format!("unknown parent default class `{parent}`"))
                    })?;
                    chain.push(class);
                }
                let mut attrs = BTreeMap::new();
                for class in chain.into_iter().rev() {
                    if let Some(values) = class.actuators.get(&kind) {
                        attrs.extend(values.clone());
                    }
                    if let Some(values) = class.actuators.get("general") {
                        attrs.extend(values.clone());
                    }
                }
                attrs.extend(explicit);
                let entry = match parse(&kind, &attrs, joints, entries.len()) {
                    Ok(entry) => entry,
                    Err(MjcfLoadError::Unsupported(reason))
                        if reason == "negative actuator damping" =>
                    {
                        return Err(MjcfLoadError::Unsupported(reason));
                    }
                    Err(MjcfLoadError::Unsupported(reason)) => MjcfActuatorInfo {
                        name: attrs
                            .get("name")
                            .cloned()
                            .unwrap_or_else(|| format!("actuator_{}", entries.len())),
                        joint: attrs.get("joint").cloned().unwrap_or_default(),
                        coordinate: attrs.get("joint").and_then(|joint| {
                            let mut targets = joints.iter().filter(|j| &j.name == joint);
                            let target = targets.next()?;
                            (targets.next().is_none()
                                && target.dofs.len() == 1
                                && matches!(
                                    target.kind,
                                    JointKind::Revolute | JointKind::Prismatic
                                ))
                            .then_some(target.dofs.start)
                        }),
                        kind: MjcfActuatorKind::Unsupported,
                        source_kind: kind,
                        gear: parse_optional_numbers(&attrs, "gear")?
                            .and_then(|values| values.first().copied())
                            .unwrap_or(1.0),
                        control_range: range(&attrs, "ctrl")?,
                        force_range: range(&attrs, "force")?,
                        attributes: attrs,
                        unsupported_reason: Some(reason),
                    },
                    Err(error) => return Err(error),
                };
                entries.push(entry);
                if starts {
                    nested += 1;
                }
            }
            Ok(Event::Start(_)) if inside => nested += 1,
            Ok(Event::End(_)) if inside => nested -= 1,
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
    if !matches!(
        kind,
        "motor" | "position" | "velocity" | "damper" | "general"
    ) {
        return Err(MjcfLoadError::Unsupported(format!(
            "actuator type `{kind}`"
        )));
    }
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
        coordinate: Some(target.dofs.start),
        kind: dynamics,
        source_kind: kind.into(),
        attributes: attrs.clone(),
        unsupported_reason: None,
        gear: gear_values[0],
        control_range: range(attrs, "ctrl")?,
        force_range: range(attrs, "force")?,
    })
}
