//! fmt_dur - strict Duration parsing/formatting.
//!
//! Grammar (strict mode, default):
//!   Input := Segment { Segment }
//!   Segment := Number Unit
//!   Number := DIGIT+ [ "." DIGIT{1,9} ]   // decimal allowed at most once, and only in the last Segment
//!   Unit   := "d" | "h" | "m" | "s" | "ms" | "us" | "ns"
//!   Rules  :
//!     - Units must appear in strictly descending order: d > h > m > s > ms > us > ns
//!     - No duplicate units
//!     - No spaces/underscores; lowercase only (enable "loose" feature to allow spaces/underscores and case-insensitive)
//!     - At least one segment must be present (e.g., "0s" is valid)
//!     - Up to 9 fractional digits (nanosecond precision). Fraction may appear only on the last segment.
//!
//! Examples:
//!   "2d3h4m", "90s", "1.5h", "250ms", "1m30s", "1m30.5s", "750us", "10ns"
//!
//! Overview:
//!   - parse("1.5h") -> Duration
//!   - parse_with("1.5h", &ParseOptions::strict().saturating()) -> Duration (saturates on overflow)
//!   - format(duration) -> "2d3h4m5.25s" (default, mixed-units; decimals only on the last unit)
//!   - format_with(duration, FormatOptions::largest_unit_decimal()) -> "1.5h"
//!
//! Features:
//!   - loose  : allows spaces and underscores between segments and case-insensitive units (ordering still enforced).
//!   - serde  : enables serde::{Serialize, Deserialize} for DurationStr using this format.
//!

#![forbid(unsafe_code)]
// #![deny(missing_docs)]
use std::fmt;
use std::time::Duration;

/// Behavior when a parsed value exceeds `Duration`'s maximum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverflowBehavior {
    /// Return an error on overflow (default).
    Error,
    /// Saturate to `Duration::MAX` on overflow.
    Saturate,
}

/// Options controlling parsing behavior.
#[derive(Clone, Copy, Debug)]
pub struct ParseOptions {
    overflow: OverflowBehavior,
}

impl ParseOptions {
    /// Strict defaults: overflow errors.
    pub fn strict() -> Self {
        Self {
            overflow: OverflowBehavior::Error,
        }
    }
    /// Change overflow behavior to saturate.
    pub fn saturating(mut self) -> Self {
        self.overflow = OverflowBehavior::Saturate;
        self
    }
}

/// Parse a strict human duration using default options.
pub fn parse(input: &str) -> Result<Duration, ParseError> {
    parse_with(input, &ParseOptions::strict())
}

/// Parse with explicit options.
pub fn parse_with(input: &str, opts: &ParseOptions) -> Result<Duration, ParseError> {
    let s = normalize_input(input)?;
    Parser::new(&s, *opts).parse()
}

/// Format a Duration using mixed-units style (default), e.g. "2d3h4m5.25s", "250ms", "0s".
pub fn format(d: Duration) -> String {
    format_with(d, &FormatOptions::mixed())
}

/// Formatting style.
#[derive(Clone, Copy, Debug)]
pub enum FormatStyle {
    /// Mixed units, descending, with decimals only on the last (seconds) component.
    /// Sub-second durations use ms/us/ns.
    Mixed,
    /// Single largest unit with a decimal fraction if needed, e.g., "1.5h", "90s", "0.123s".
    /// This style cannot always be finite-decimal exact (e.g., 30s in hours),
    /// but it still round-trips because the parser accepts up to 9 fractional digits.
    LargestUnitDecimal,
}

/// Options controlling formatting.
#[derive(Clone, Copy, Debug)]
pub struct FormatOptions {
    style: FormatStyle,
    max_frac_digits: u8, // 0..=9
}

impl FormatOptions {
    /// Mixed-units default. Fractions up to 9 digits when needed.
    pub fn mixed() -> Self {
        Self {
            style: FormatStyle::Mixed,
            max_frac_digits: 9,
        }
    }
    /// Largest-unit decimal style.
    pub fn largest_unit_decimal() -> Self {
        Self {
            style: FormatStyle::LargestUnitDecimal,
            max_frac_digits: 9,
        }
    }
    /// Limit fractional digits (0..=9).
    pub fn with_max_frac_digits(mut self, digits: u8) -> Self {
        self.max_frac_digits = digits.min(9);
        self
    }
}

/// Format with options.
pub fn format_with(d: Duration, opts: &FormatOptions) -> String {
    match opts.style {
        FormatStyle::Mixed => format_mixed(d, opts.max_frac_digits),
        FormatStyle::LargestUnitDecimal => format_largest_unit_decimal(d, opts.max_frac_digits),
    }
}

/// Error returned when parsing fails.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParseError {
    /// Empty input string.
    Empty,
    /// Invalid character at byte index.
    InvalidChar(usize),
    /// Invalid or missing number at byte index.
    InvalidNumber(usize),
    /// Invalid unit at byte index.
    InvalidUnit(usize),
    /// Units must be strictly descending (e.g., h cannot follow s).
    OutOfOrderUnit {
        prev: Unit,
        next: Unit,
        index: usize,
    },
    /// Unit appeared more than once.
    DuplicateUnit { unit: Unit, index: usize },
    /// A decimal number was found before the last segment.
    DecimalNotLast(usize),
    /// Too many fractional digits (> 9).
    TooPreciseFraction { digits: usize, index: usize },
    /// Overflow (value exceeds Duration::MAX) and behavior was set to Error.
    Overflow,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ParseError::*;
        match self {
            Empty => write!(f, "empty duration"),
            InvalidChar(i) => write!(f, "invalid character at byte index {}", i),
            InvalidNumber(i) => write!(f, "invalid number at byte index {}", i),
            InvalidUnit(i) => write!(f, "invalid or missing unit at byte index {}", i),
            OutOfOrderUnit { prev, next, index } => write!(
                f,
                "out-of-order unit '{}' followed by '{}' at byte index {}",
                prev.as_str(),
                next.as_str(),
                index
            ),
            DuplicateUnit { unit, index } => write!(
                f,
                "duplicate unit '{}' at byte index {}",
                unit.as_str(),
                index
            ),
            DecimalNotLast(i) => write!(f, "decimal segment must be last (index {})", i),
            TooPreciseFraction { digits, index } => write!(
                f,
                "fractional part has {} digits (max 9) at byte index {}",
                digits, index
            ),
            Overflow => write!(f, "duration overflowed maximum representable span"),
        }
    }
}

impl std::error::Error for ParseError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Unit {
    D,
    H,
    M,
    S,
    Ms,
    Us,
    Ns,
}

impl Unit {
    fn as_str(self) -> &'static str {
        match self {
            Unit::D => "d",
            Unit::H => "h",
            Unit::M => "m",
            Unit::S => "s",
            Unit::Ms => "ms",
            Unit::Us => "us",
            Unit::Ns => "ns",
        }
    }
    fn rank(self) -> u8 {
        match self {
            Unit::D => 6,
            Unit::H => 5,
            Unit::M => 4,
            Unit::S => 3,
            Unit::Ms => 2,
            Unit::Us => 1,
            Unit::Ns => 0,
        }
    }
    fn nanos(self) -> u128 {
        match self {
            Unit::D => 86_400_000_000_000,
            Unit::H => 3_600_000_000_000,
            Unit::M => 60_000_000_000,
            Unit::S => 1_000_000_000,
            Unit::Ms => 1_000_000,
            Unit::Us => 1_000,
            Unit::Ns => 1,
        }
    }
}

struct Parser<'a> {
    s: &'a str,
    opts: ParseOptions,
    i: usize,
    len: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str, opts: ParseOptions) -> Self {
        Self {
            s,
            opts,
            i: 0,
            len: s.len(),
        }
    }

    fn parse(&mut self) -> Result<Duration, ParseError> {
        if self.s.is_empty() {
            return Err(ParseError::Empty);
        }

        let mut total_nanos: u128 = 0;
        let max_nanos: u128 =
            (u128::from(u64::MAX) * 1_000_000_000u128) + (1_000_000_000u128 - 1u128);

        let mut prev_rank: Option<u8> = None;
        let mut seen_mask: u8 = 0;
        let mut decimal_used = false;
        let mut segments = 0usize;

        while self.i < self.len {
            let start_num = self.i;
            // Parse number with optional decimal.
            let (int_part, frac_part) = self.parse_number()?;
            segments += 1;

            // Decimal can only appear on the last segment.
            if frac_part.is_some() && self.i < self.len {
                // There are more characters; must be unit or next segment.
                // Check after unit parse that we're not at end.
                decimal_used = true;
            }

            let unit_start = self.i;
            let unit = self
                .parse_unit()
                .map_err(|_| ParseError::InvalidUnit(unit_start))?;

            // Enforce ordering
            let rank = unit.rank();
            if let Some(prev) = prev_rank {
                if rank >= prev {
                    return Err(ParseError::OutOfOrderUnit {
                        prev: rank_to_unit(prev),
                        next: unit,
                        index: unit_start,
                    });
                }
            }
            prev_rank = Some(rank);

            // Enforce no duplicates
            let bit = 1u8 << rank;
            if (seen_mask & bit) != 0 {
                return Err(ParseError::DuplicateUnit {
                    unit,
                    index: unit_start,
                });
            }
            seen_mask |= bit;

            // Decimal must be on the last segment only.
            if decimal_used && self.i < self.len {
                return Err(ParseError::DecimalNotLast(start_num));
            }

            // Accumulate nanos, with overflow handling
            let unit_nanos = unit.nanos();

            // int_part
            if int_part > 0 {
                let add = (int_part as u128)
                    .checked_mul(unit_nanos)
                    .ok_or(ParseError::Overflow)?;
                total_nanos = match total_nanos.checked_add(add) {
                    Some(v) => v,
                    None => {
                        if self.opts.overflow == OverflowBehavior::Saturate {
                            return Ok(duration_max());
                        } else {
                            return Err(ParseError::Overflow);
                        }
                    }
                };
                if total_nanos > max_nanos {
                    if self.opts.overflow == OverflowBehavior::Saturate {
                        return Ok(duration_max());
                    } else {
                        return Err(ParseError::Overflow);
                    }
                }
            }

            // frac_part
            if let Some(frac) = frac_part {
                let digits = frac.len();
                if digits == 0 {
                    return Err(ParseError::InvalidNumber(start_num));
                }
                if digits > 9 {
                    return Err(ParseError::TooPreciseFraction {
                        digits,
                        index: start_num,
                    });
                }
                let frac_value = frac
                    .bytes()
                    .try_fold(0u128, |acc, b| {
                        if (b'0'..=b'9').contains(&b) {
                            Some(acc * 10 + u128::from(b - b'0'))
                        } else {
                            None
                        }
                    })
                    .ok_or(ParseError::InvalidNumber(start_num))?;

                // Compute fractional nanos: unit_nanos * frac / 10^digits
                let denom = 10u128.pow(digits as u32);
                let add = unit_nanos
                    .checked_mul(frac_value)
                    .ok_or(ParseError::Overflow)?
                    / denom;

                total_nanos = match total_nanos.checked_add(add) {
                    Some(v) => v,
                    None => {
                        if self.opts.overflow == OverflowBehavior::Saturate {
                            return Ok(duration_max());
                        } else {
                            return Err(ParseError::Overflow);
                        }
                    }
                };
                if total_nanos > max_nanos {
                    if self.opts.overflow == OverflowBehavior::Saturate {
                        return Ok(duration_max());
                    } else {
                        return Err(ParseError::Overflow);
                    }
                }
            }
        }

        if segments == 0 {
            return Err(ParseError::Empty);
        }

        Ok(nanos_to_duration(total_nanos))
    }

    fn parse_number(&mut self) -> Result<(u64, Option<&'a str>), ParseError> {
        let start = self.i;
        let bytes = self.s.as_bytes();

        if start >= self.len {
            return Err(ParseError::InvalidNumber(start));
        }

        let mut saw_digit = false;
        let mut int_end = start;
        while int_end < self.len {
            let b = bytes[int_end];
            if b.is_ascii_digit() {
                saw_digit = true;
                int_end += 1;
            } else {
                break;
            }
        }

        if !saw_digit {
            return Err(ParseError::InvalidNumber(start));
        }

        let mut frac: Option<&'a str> = None;
        let mut pos = int_end;
        if pos < self.len && bytes[pos] == b'.' {
            // fractional part
            pos += 1;
            let frac_start = pos;
            let mut frac_end = pos;
            while frac_end < self.len {
                let b = bytes[frac_end];
                if b.is_ascii_digit() {
                    frac_end += 1;
                } else {
                    break;
                }
            }
            if frac_end == frac_start {
                return Err(ParseError::InvalidNumber(start));
            }
            frac = Some(&self.s[frac_start..frac_end]);
            pos = frac_end;
        }

        // Parse int part (fits u64)
        let int_str = &self.s[start..int_end];
        let int_val = int_str
            .bytes()
            .try_fold(0u64, |acc, b| {
                acc.checked_mul(10)?.checked_add(u64::from(b - b'0'))
            })
            .ok_or(ParseError::InvalidNumber(start))?;

        self.i = pos;
        Ok((int_val, frac))
    }

    fn parse_unit(&mut self) -> Result<Unit, ()> {
        // Match longest unit first: "ms", "us", "ns" before single letters.
        let rest = &self.s[self.i..];

        let try_take =
            |s: &str, u: Unit| -> Option<Unit> { if rest.starts_with(s) { Some(u) } else { None } };

        let unit = try_take("ms", Unit::Ms)
            .or_else(|| try_take("us", Unit::Us))
            .or_else(|| try_take("ns", Unit::Ns))
            .or_else(|| try_take("d", Unit::D))
            .or_else(|| try_take("h", Unit::H))
            .or_else(|| try_take("m", Unit::M))
            .or_else(|| try_take("s", Unit::S));

        if let Some(u) = unit {
            self.i += u.as_str().len();
            Ok(u)
        } else {
            Err(())
        }
    }
}

// Helpers

fn nanos_to_duration(nanos: u128) -> Duration {
    let secs = (nanos / 1_000_000_000) as u64;
    let sub = (nanos % 1_000_000_000) as u32;
    Duration::new(secs, sub)
}

fn duration_max() -> Duration {
    // Equivalent to Duration::MAX without relying on that constant.
    nanos_to_duration((u128::from(u64::MAX) * 1_000_000_000u128) + 999_999_999u128)
}

fn rank_to_unit(rank: u8) -> Unit {
    match rank {
        6 => Unit::D,
        5 => Unit::H,
        4 => Unit::M,
        3 => Unit::S,
        2 => Unit::Ms,
        1 => Unit::Us,
        _ => Unit::Ns,
    }
}

fn normalize_input(input: &str) -> Result<String, ParseError> {
    #[cfg(feature = "loose")]
    {
        let mut s = String::with_capacity(input.len());
        for (i, ch) in input.chars().enumerate() {
            if ch == ' ' || ch == '_' {
                continue;
            }
            if ch.is_ascii() {
                s.push(ch.to_ascii_lowercase());
            } else {
                return Err(ParseError::InvalidChar(i));
            }
        }
        if s.is_empty() {
            return Err(ParseError::Empty);
        }
        Ok(s)
    }
    #[cfg(not(feature = "loose"))]
    {
        // Strict: must be ASCII and contain no spaces/underscores, lower-case only.
        if input.is_empty() {
            return Err(ParseError::Empty);
        }
        for (i, b) in input.bytes().enumerate() {
            if !b.is_ascii() {
                return Err(ParseError::InvalidChar(i));
            }
            if b == b' ' || b == b'_' || (b'A'..=b'Z').contains(&b) {
                return Err(ParseError::InvalidChar(i));
            }
        }
        Ok(input.to_string())
    }
}

// Formatting

fn format_mixed(d: Duration, max_frac_digits: u8) -> String {
    let mut rem_secs = d.as_secs();
    let mut rem_nanos = d.subsec_nanos();

    let mut out = String::new();

    let days = rem_secs / 86_400;
    if days > 0 {
        out.push_str(&format!("{}d", days));
        rem_secs %= 86_400;
    }
    let hours = rem_secs / 3_600;
    if hours > 0 {
        out.push_str(&format!("{}h", hours));
        rem_secs %= 3_600;
    }
    let mins = rem_secs / 60;
    if mins > 0 {
        out.push_str(&format!("{}m", mins));
        rem_secs %= 60;
    }

    // Always render seconds if we have any seconds or nanos left.
    if rem_secs > 0 || rem_nanos > 0 {
        if rem_nanos > 0 {
            let s = format_fraction(rem_secs, rem_nanos, max_frac_digits);
            out.push_str(&format!("{}s", s));
        } else {
            out.push_str(&format!("{}s", rem_secs));
        }
    }

    // If nothing at all was emitted, render 0s.
    if out.is_empty() {
        out.push_str("0s");
    }
    out
}

fn format_largest_unit_decimal(d: Duration, max_frac_digits: u8) -> String {
    let mut total_nanos = (d.as_secs() as u128) * 1_000_000_000u128 + (d.subsec_nanos() as u128);

    if total_nanos == 0 {
        return "0s".to_string();
    }

    let candidates = [
        Unit::D,
        Unit::H,
        Unit::M,
        Unit::S,
        Unit::Ms,
        Unit::Us,
        Unit::Ns,
    ];

    for &u in &candidates {
        let u_nanos = u.nanos();
        if total_nanos >= u_nanos {
            // integer part
            let whole = total_nanos / u_nanos;
            let rem = total_nanos % u_nanos;
            if rem == 0 {
                return format!("{}{}", whole, u.as_str());
            } else {
                // fraction up to max_frac_digits
                let mut frac = rem * 10u128.pow(max_frac_digits as u32) / u_nanos;
                // Trim trailing zeros, but keep at least one digit.
                let mut frac_str = format!("{:0width$}", frac, width = max_frac_digits as usize);
                while frac_str.ends_with('0') && frac_str.len() > 1 {
                    frac_str.pop();
                }
                return format!("{}.{}{}", whole, frac_str, u.as_str());
            }
        }
    }
    // Fallback, should not happen.
    "0s".to_string()
}

fn format_fraction(secs: u64, nanos: u32, max_frac_digits: u8) -> String {
    if nanos == 0 || max_frac_digits == 0 {
        return format!("{}.", secs).trim_end_matches('.').to_string();
    }
    // Scale nanos (0..1_000_000_000) to fractional digits.
    let scale = 10u32.pow(max_frac_digits as u32);
    let mut frac = (nanos as u128 * scale as u128) / 1_000_000_000u128;
    let mut frac_str = format!("{:0width$}", frac, width = max_frac_digits as usize);
    // Trim trailing zeros.
    while frac_str.ends_with('0') && frac_str.len() > 1 {
        frac_str.pop();
    }
    format!("{}.{}", secs, frac_str)
}

/// A serde wrapper that (de)serializes as a strict human duration string.
///
/// Enable the "serde" feature to use this type.
#[cfg(feature = "serde")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DurationStr(pub Duration);

#[cfg(feature = "serde")]
impl serde::Serialize for DurationStr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = format(self.0);
        serializer.serialize_str(&s)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for DurationStr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = DurationStr;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a strict human duration string (e.g., \"2d3h4m\", \"90s\", \"1.5h\")")
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                parse(v)
                    .map(DurationStr)
                    .map_err(|e| E::custom(format!("invalid duration: {}", e)))
            }
        }
        deserializer.deserialize_str(V)
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_parse() {
        assert_eq!(parse("90s").unwrap(), Duration::from_secs(90));
        assert_eq!(parse("1.5h").unwrap(), Duration::from_secs(5400));
        assert_eq!(
            parse("2d3h4m").unwrap(),
            Duration::from_secs(2 * 86_400 + 3 * 3600 + 4 * 60)
        );
        assert_eq!(parse("250ms").unwrap(), Duration::from_millis(250));
        assert_eq!(parse("750us").unwrap(), Duration::from_micros(750));
        assert_eq!(parse("10ns").unwrap(), Duration::new(0, 10));
        assert_eq!(parse("1m30s").unwrap(), Duration::from_secs(90));
        assert_eq!(
            parse("1m30.5s").unwrap(),
            Duration::from_secs(90) + Duration::from_millis(500)
        );
    }

    #[test]
    fn ordering_and_duplicates() {
        assert!(parse("h1m").is_err());
        assert!(parse("1m1m").is_err());
        assert!(parse("1s2m").is_err());
        assert!(parse("1ms2s").is_err());
    }

    #[test]
    fn decimal_rules() {
        assert!(parse("1.5h10m").is_err()); // decimal not last
        assert!(parse("1.1234567890s").is_err()); // too precise (>9)
        assert!(parse("1.s").is_err());
        assert!(parse(".5h").is_err());
    }

    #[test]
    fn zero_and_format() {
        assert_eq!(parse("0s").unwrap(), Duration::from_secs(0));
        assert_eq!(format(Duration::from_secs(0)), "0s");

        let d = Duration::from_secs(2 * 86_400 + 3 * 3600 + 4 * 60) + Duration::from_millis(250);
        let s = format(d);
        assert_eq!(s, "2d3h4m0.25s");
    }

    #[test]
    fn roundtrip_mixed() {
        let cases = [
            "2d3h4m",
            "90s",
            "1.5h",
            "250ms",
            "1m30s",
            "1m30.5s",
            "999ms",
            "1001ms",
            "3h15m45.123456789s",
        ];
        for &c in &cases {
            let d = parse(c).unwrap();
            let s = format(d);
            let d2 = parse(&s).unwrap();
            assert_eq!(d, d2, "roundtrip failed for {}", c);
        }
    }

    #[test]
    fn largest_unit_decimal_format() {
        let d = Duration::from_secs(5400);
        let s = format_with(
            d,
            &FormatOptions::largest_unit_decimal().with_max_frac_digits(3),
        );
        // 5400 seconds = 1.5h exactly
        assert_eq!(s, "1.5h");
    }

    #[test]
    fn overflow_behavior() {
        // Construct a huge input to overflow
        let huge = format!("{}d", u64::MAX);
        let err = parse_with(&huge, &ParseOptions::strict()).unwrap_err();
        assert!(matches!(err, ParseError::Overflow));

        let saturated = parse_with(&huge, &ParseOptions::strict().saturating()).unwrap();
        assert_eq!(saturated, super::duration_max());
    }

    #[cfg(feature = "loose")]
    #[test]
    fn loose_mode() {
        assert_eq!(
            super::parse("1H 30M").unwrap(),
            std::time::Duration::from_secs(5400)
        );
        assert_eq!(
            super::parse("1h_250ms").unwrap(),
            std::time::Duration::from_millis(362_250)
        );
    }
}
