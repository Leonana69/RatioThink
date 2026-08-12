import XCTest
@testable import RatioThinkCore

/// `Profile.nextTokenArena` convention tests: a profile dispatches as Next
/// Token Arena iff `inferlet_args.mode == "next-token-arena"`. The launched
/// inferlet stays `chat-apc`; the chat send path turns this mode into a
/// per-request gateway route.
final class NextTokenArenaProfileTests: XCTestCase {

  private func parse(_ toml: String) throws -> Profile { try Profile.parse(toml: toml) }

  func test_non_nextTokenArena_profile_is_nil() throws {
    let profile = try parse("""
    id = "chat"
    name = "Chat"
    model = "qwen"
    inferlet = "chat-apc"
    """)

    XCTAssertNil(profile.nextTokenArena)
  }

  func test_other_mode_value_is_nil() throws {
    let profile = try parse("""
    id = "x"
    name = "X"
    model = "qwen"
    inferlet = "chat-apc"

    [inferlet_args]
    mode = "best-of-n"
    """)

    XCTAssertNil(profile.nextTokenArena)
  }

  func test_nextTokenArena_profile_reads_mode() throws {
    let profile = try parse("""
    id = "next-token-arena"
    name = "Next Token Arena"
    model = "qwen"
    inferlet = "chat-apc"

    [inferlet_args]
    mode = "next-token-arena"
    """)

    XCTAssertEqual(profile.nextTokenArena, NextTokenArenaProfileConfig())
    XCTAssertEqual(profile.inferlet, "chat-apc")
  }

  func test_nextTokenArena_profile_round_trips_through_dump() throws {
    let profile = try parse("""
    id = "next-token-arena"
    name = "Next Token Arena"
    model = "qwen"
    inferlet = "chat-apc"

    [inferlet_args]
    mode = "next-token-arena"
    """)

    let reparsed = try Profile.parse(toml: try profile.dump())
    XCTAssertEqual(reparsed.nextTokenArena, NextTokenArenaProfileConfig())
  }
}
