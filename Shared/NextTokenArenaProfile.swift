import Foundation
import TOMLKit

public struct NextTokenArenaProfileConfig: Equatable, Sendable {
  public init() {}
}

public extension Profile {
  /// `inferlet_args.mode` value that turns a profile into the Next Token Arena
  /// game. Like ToT and Best-of-N, this is a per-request dispatch mode; the
  /// launch-time inferlet remains `chat-apc`, while the launch resolver forces
  /// gateway mode so requests can route to the `next-token-arena` wasm.
  static let nextTokenArenaModeValue = "next-token-arena"

  var nextTokenArena: NextTokenArenaProfileConfig? {
    guard inferletArgs[Profile.dispatchModeArgKey]?.tomlValue.string == Profile.nextTokenArenaModeValue
    else {
      return nil
    }
    return NextTokenArenaProfileConfig()
  }
}
