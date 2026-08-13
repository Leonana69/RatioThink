import SwiftUI

struct ProbabilityLensCard: View {
  let payload: ProbabilityLensPayload
  let maxWidth: CGFloat

  @State private var selectedTokenIndex: Int?

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack(spacing: 8) {
        Image(systemName: "eye")
          .foregroundStyle(Color.accentColor)
        Text("Probability Lens")
          .font(.headline)
        Spacer(minLength: 12)
        if let entropy = payload.averageEntropy {
          Text("H \(format(entropy))")
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
        }
      }

      HStack(spacing: 10) {
        legendSwatch(color: .green, label: "Confident")
        legendSwatch(color: .yellow, label: "Ambiguous")
        legendSwatch(color: .red, label: "Uncertain")
      }

      if payload.tokens.isEmpty {
        Text(payload.text)
          .textSelection(.enabled)
          .frame(maxWidth: .infinity, alignment: .leading)
      } else {
        ProbabilityTokenFlowLayout(spacing: 1) {
          ForEach(Array(payload.tokens.enumerated()), id: \.offset) { index, token in
            tokenButton(token, at: index)
          }
        }
      }
    }
    .padding(12)
    .frame(maxWidth: maxWidth, alignment: .leading)
    .background(Color.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
    .overlay(
      RoundedRectangle(cornerRadius: 8)
        .strokeBorder(Color.secondary.opacity(0.16))
    )
    .accessibilityIdentifier("message.probabilityLens")
  }

  private func tokenButton(
    _ token: ProbabilityLensPayload.TokenSpan,
    at index: Int
  ) -> some View {
    Button {
      selectedTokenIndex = index
    } label: {
      Text(displayToken(token.text))
        .font(.body.monospaced())
        .foregroundStyle(.primary)
        .padding(.horizontal, 2)
        .padding(.vertical, 1)
        .background(confidenceColor(token.entropy).opacity(0.22))
        .contentShape(Rectangle())
    }
    .buttonStyle(.plain)
    .help(tokenHelp(token))
    .accessibilityLabel(tokenAccessibilityLabel(token))
    .popover(isPresented: selectionBinding(for: index), arrowEdge: .bottom) {
      ProbabilityTokenDetail(token: token)
    }
  }

  private func selectionBinding(for index: Int) -> Binding<Bool> {
    Binding(
      get: { selectedTokenIndex == index },
      set: { isPresented in
        if !isPresented, selectedTokenIndex == index { selectedTokenIndex = nil }
      }
    )
  }

  private func legendSwatch(color: Color, label: String) -> some View {
    HStack(spacing: 4) {
      Rectangle()
        .fill(color.opacity(0.55))
        .frame(width: 8, height: 8)
      Text(label)
        .font(.caption2)
        .foregroundStyle(.secondary)
    }
  }

  private func confidenceColor(_ entropy: Double?) -> Color {
    guard let entropy else { return .secondary }
    if entropy < 1.5 { return .green }
    if entropy < 3.0 { return .yellow }
    return .red
  }

  private func tokenHelp(_ token: ProbabilityLensPayload.TokenSpan) -> String {
    var parts: [String] = []
    if let probability = token.probability {
      parts.append("Selected \(percent(probability))")
    }
    if let entropy = token.entropy {
      parts.append("Entropy \(format(entropy))")
    }
    return parts.isEmpty ? "Inspect alternatives" : parts.joined(separator: ", ")
  }

  private func tokenAccessibilityLabel(_ token: ProbabilityLensPayload.TokenSpan) -> String {
    "\(displayToken(token.text)), \(tokenHelp(token))"
  }

  private func displayToken(_ raw: String) -> String {
    let visible = raw
      .replacingOccurrences(of: "\n", with: "\\n")
      .replacingOccurrences(of: "\t", with: "\\t")
      .replacingOccurrences(of: "\r", with: "\\r")
    return visible.isEmpty ? "<empty>" : visible
  }

  private func percent(_ value: Double) -> String {
    String(format: "%.1f%%", value * 100)
  }

  private func format(_ value: Double) -> String {
    String(format: "%.2f", value)
  }
}

private struct ProbabilityTokenDetail: View {
  let token: ProbabilityLensPayload.TokenSpan

  private var maxProbability: Double {
    max(token.alternatives.map(\.probability).max() ?? 0.001, 0.001)
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack(alignment: .firstTextBaseline) {
        Text(displayToken(token.text))
          .font(.headline.monospaced())
        Spacer(minLength: 20)
        if let probability = token.probability {
          Text(percent(probability))
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
        }
      }
      if let entropy = token.entropy {
        Text("Entropy \(String(format: "%.2f", entropy))")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      Divider()
      ForEach(token.alternatives) { alternative in
        VStack(alignment: .leading, spacing: 3) {
          HStack(spacing: 12) {
            Text(displayToken(alternative.text))
              .font(.caption.monospaced())
              .lineLimit(1)
            Spacer(minLength: 12)
            Text(percent(alternative.probability))
              .font(.caption.monospacedDigit())
              .foregroundStyle(.secondary)
          }
          GeometryReader { proxy in
            Rectangle()
              .fill(alternative.id == token.id ? Color.accentColor : Color.secondary.opacity(0.45))
              .frame(width: proxy.size.width * min(1, alternative.probability / maxProbability))
          }
          .frame(height: 3)
        }
      }
    }
    .padding(12)
    .frame(width: 280)
  }

  private func displayToken(_ raw: String) -> String {
    let visible = raw
      .replacingOccurrences(of: "\n", with: "\\n")
      .replacingOccurrences(of: "\t", with: "\\t")
      .replacingOccurrences(of: "\r", with: "\\r")
    return visible.isEmpty ? "<empty>" : visible
  }

  private func percent(_ value: Double) -> String {
    String(format: "%.1f%%", value * 100)
  }
}

private struct ProbabilityTokenFlowLayout: Layout {
  var spacing: CGFloat

  func sizeThatFits(
    proposal: ProposedViewSize,
    subviews: Subviews,
    cache: inout ()
  ) -> CGSize {
    layout(proposal: proposal, subviews: subviews).size
  }

  func placeSubviews(
    in bounds: CGRect,
    proposal: ProposedViewSize,
    subviews: Subviews,
    cache: inout ()
  ) {
    let result = layout(
      proposal: ProposedViewSize(width: bounds.width, height: proposal.height),
      subviews: subviews)
    for (index, point) in result.points.enumerated() {
      let size = result.sizes[index]
      subviews[index].place(
        at: CGPoint(x: bounds.minX + point.x, y: bounds.minY + point.y),
        anchor: .topLeading,
        proposal: ProposedViewSize(width: size.width, height: size.height))
    }
  }

  private func layout(
    proposal: ProposedViewSize,
    subviews: Subviews
  ) -> (size: CGSize, points: [CGPoint], sizes: [CGSize]) {
    let availableWidth = max(1, proposal.width ?? 600)
    var points: [CGPoint] = []
    var sizes: [CGSize] = []
    var x: CGFloat = 0
    var y: CGFloat = 0
    var lineHeight: CGFloat = 0

    for subview in subviews {
      let ideal = subview.sizeThatFits(.unspecified)
      let size = ideal.width <= availableWidth
        ? ideal
        : subview.sizeThatFits(ProposedViewSize(width: availableWidth, height: nil))
      if x > 0, x + size.width > availableWidth {
        x = 0
        y += lineHeight + spacing
        lineHeight = 0
      }
      points.append(CGPoint(x: x, y: y))
      sizes.append(size)
      x += size.width + spacing
      lineHeight = max(lineHeight, size.height)
    }

    return (CGSize(width: availableWidth, height: y + lineHeight), points, sizes)
  }
}
