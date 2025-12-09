import './Header.css'

function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo-section">
          <div className="recycle-icon">♻️</div>
          <div className="header-text">
            <h1>Waste Material Identifier</h1>
            <p className="subtitle">AI-Powered Waste Classification System</p>
          </div>
        </div>
        <div className="supported-materials">
          <span className="material-badge glass">Glass</span>
          <span className="material-badge paper">Paper</span>
          <span className="material-badge cardboard">Cardboard</span>
          <span className="material-badge plastic">Plastic</span>
          <span className="material-badge metal">Metal</span>
          <span className="material-badge trash">Trash</span>
        </div>
      </div>
    </header>
  )
}

export default Header
