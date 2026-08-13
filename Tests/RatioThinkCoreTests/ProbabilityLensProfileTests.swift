import XCTest
@testable import RatioThinkCore

final class ProbabilityLensProfileTests: XCTestCase {
  private func parse(_ toml: String) throws -> Profile { try Profile.parse(toml: toml) }

  func test_profile_requires_probabilityLens_mode() throws {
    let chat = try parse("""
    id = "chat"
    name = "Chat"
    model = "qwen"
    inferlet = "chat-apc"
    """)
    XCTAssertNil(chat.probabilityLens)

    let lens = try parse("""
    id = "probability-lens"
    name = "Probability Lens"
    model = "qwen"
    inferlet = "chat-apc"

    [inferlet_args]
    mode = "probability-lens"
    """)
    XCTAssertEqual(lens.probabilityLens, ProbabilityLensProfileConfig())
    XCTAssertEqual(lens.inferlet, "chat-apc")
  }

  func test_builtin_round_trips() throws {
    let profile = try parse(ProfileStore.probabilityLensTOML)
    XCTAssertEqual(profile.id, ProfileStore.probabilityLensProfileID)
    XCTAssertEqual(profile.probabilityLens, ProbabilityLensProfileConfig())

    let reparsed = try Profile.parse(toml: try profile.dump())
    XCTAssertEqual(reparsed.probabilityLens, ProbabilityLensProfileConfig())
  }

  func test_payload_decode_and_plain_text() throws {
    let json = """
    {
      "kind":"probability_lens",
      "text":"A door",
      "tokens":[{
        "id":10,
        "text":" door",
        "probability":0.42,
        "entropy":1.2,
        "alternatives":[{"id":10,"text":" door","probability":0.42}]
      }],
      "average_entropy":1.2,
      "temperature":0.7,
      "top_p":0.9
    }
    """
    let payload = try XCTUnwrap(ProbabilityLensPayload.decode(from: json))
    XCTAssertEqual(payload.text, "A door")
    XCTAssertEqual(payload.tokens.first?.alternatives.first?.probability, 0.42)
    XCTAssertNil(ProbabilityLensPayload.decode(from: "{\"kind\":\"other\"}"))
  }
}
