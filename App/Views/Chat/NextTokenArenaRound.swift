import Foundation

struct NextTokenArenaRound: Codable, Equatable {
  var kind: String
  var prompt: String
  var picked: Token
  var top: [Token]
  var branches: [BranchPreview]
  var entropy: Double
  var temperature: Double
  var topP: Double

  var isArenaPayload: Bool {
    kind == "next_token_arena"
  }

  static func decode(from content: String) -> NextTokenArenaRound? {
    guard let data = content.data(using: .utf8),
          let round = try? JSONDecoder().decode(NextTokenArenaRound.self, from: data),
          round.isArenaPayload else {
      return nil
    }
    return round
  }

  struct Token: Codable, Equatable, Identifiable {
    var id: UInt32
    var text: String
    var probability: Double
  }

  struct BranchPreview: Codable, Equatable, Identifiable {
    var token: Token
    var preview: String
    var generatedTokens: Int

    var id: UInt32 {
      token.id
    }

    private enum CodingKeys: String, CodingKey {
      case token
      case preview
      case generatedTokens = "generated_tokens"
    }
  }

  private enum CodingKeys: String, CodingKey {
    case kind
    case prompt
    case picked
    case top
    case branches
    case entropy
    case temperature
    case topP = "top_p"
  }
}
