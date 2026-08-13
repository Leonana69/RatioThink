import Foundation
import TOMLKit

public struct ProbabilityLensProfileConfig: Equatable, Sendable {
  public init() {}
}

public extension Profile {
  /// `inferlet_args.mode` value that routes chat-shaped requests through the
  /// Probability Lens gateway inferlet.
  static let probabilityLensModeValue = "probability-lens"

  var probabilityLens: ProbabilityLensProfileConfig? {
    guard inferletArgs[Profile.dispatchModeArgKey]?.tomlValue.string
            == Profile.probabilityLensModeValue else {
      return nil
    }
    return ProbabilityLensProfileConfig()
  }
}

/// Persisted payload returned by the Probability Lens inferlet. This lives in
/// Shared because request-history construction must recover `text` instead of
/// sending the telemetry JSON back to the model on the next turn.
public struct ProbabilityLensPayload: Codable, Equatable, Sendable {
  public var kind: String
  public var text: String
  public var tokens: [TokenSpan]
  public var averageEntropy: Double?
  public var temperature: Double
  public var topP: Double

  public var isProbabilityLens: Bool { kind == "probability_lens" }

  public static func decode(from content: String) -> ProbabilityLensPayload? {
    guard let data = content.data(using: .utf8),
          let payload = try? JSONDecoder().decode(Self.self, from: data),
          payload.isProbabilityLens else {
      return nil
    }
    return payload
  }

  public struct TokenSpan: Codable, Equatable, Sendable {
    public var id: UInt32
    public var text: String
    public var probability: Double?
    public var entropy: Double?
    public var alternatives: [Alternative]
  }

  public struct Alternative: Codable, Equatable, Sendable, Identifiable {
    public var id: UInt32
    public var text: String
    public var probability: Double
  }

  private enum CodingKeys: String, CodingKey {
    case kind
    case text
    case tokens
    case averageEntropy = "average_entropy"
    case temperature
    case topP = "top_p"
  }
}
