import React, { useState, useEffect } from 'react';

const Sidebar = ({ setPage }) => {
  return (
    <div style={sidebarStyle}>
      <div style={{ ...sidebarHeader, paddingBottom: '10px' }}>
        <img src="/logofront.png" alt="DeepShield Logo" style={{ width: '100%', maxHeight: '80px', objectFit: 'contain' }} />
      </div>

      <ul style={sidebarList}>
        <li style={sidebarItem} onClick={() => setPage('whitelist')}>
          <div style={sidebarIconContainer}>
            <img src="/whitelist.png" alt="Whitelist" style={iconStyle} />
            <span>Whitelist</span>
          </div>
        </li>
        <li style={sidebarItem} onClick={() => setPage('blacklist')}>
          <div style={sidebarIconContainer}>
            <img src="/blacklist.png" alt="Blacklist" style={iconStyle} />
            <span>Blacklist</span>
          </div>
        </li>
        <li style={sidebarItem} onClick={() => setPage('Sample1')}>
          <div style={sidebarIconContainer}>
            <span>Sample1</span>
          </div>
        </li>
        <li style={sidebarItem} onClick={() => setPage('Sample11')}>
          <div style={sidebarIconContainer}>
            <span>Sample11</span>
          </div>
        </li>
        <li style={sidebarItem} onClick={() => setPage('reports')}>
          <div style={sidebarIconContainer}>
            <span>Reports</span>
          </div>
        </li>
      </ul>
    </div>
  );
};

function App() {
  const [page, setPage] = useState('login');
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [users, setUsers] = useState([]);
  const [currentUser, setCurrentUser] = useState(null);

  useEffect(() => {
    const savedUsers = JSON.parse(localStorage.getItem('users')) || [];
    setUsers(savedUsers);

    const savedSession = JSON.parse(localStorage.getItem('session'));
    if (savedSession?.isLoggedIn && savedSession?.user) {
      setIsLoggedIn(true);
      setCurrentUser(savedSession.user);
      setPage('dashboard');
    }
  }, []);

  useEffect(() => {
    if (isLoggedIn && currentUser) {
      localStorage.setItem('session', JSON.stringify({ isLoggedIn: true, user: currentUser }));
    } else {
      localStorage.removeItem('session');
    }
  }, [isLoggedIn, currentUser]);

  const saveUsers = (newUsers) => {
    localStorage.setItem('users', JSON.stringify(newUsers));
    setUsers(newUsers);
  };

  const validateEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  const validatePassword = (password) => /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$/.test(password);

  const handleLogin = (email, password) => {
    if (email === 'admin' && password === 'admin') {
      const adminUser = { email: 'admin', password: 'admin', username: 'Administrator' };
      setCurrentUser(adminUser);
      setIsLoggedIn(true);
      setPage('dashboard');
      return;
    }

    const user = users.find(u => u.email === email && u.password === password);
    if (user) {
      setCurrentUser(user);
      setIsLoggedIn(true);
      setPage('dashboard');
    } else {
      alert('Invalid email or password.');
    }
  };

  const handleRegister = (email, password, confirm) => {
    if (!validateEmail(email)) return alert('Invalid email format.');
    if (!validatePassword(password)) return alert('Password must be at least 8 characters, contain a number, uppercase and lowercase letter.');
    if (password !== confirm) return alert('Passwords do not match.');
    if (users.find(u => u.email === email)) return alert('Email already registered.');
    const newUsers = [...users, { email, password }];
    saveUsers(newUsers);
    alert('Account created! You can now log in.');
    setPage('login');
  };

  const handleChangePassword = (current, newPass, confirmPass) => {
    if (currentUser.email === 'admin') return alert('Admin credentials cannot be changed.');
    if (currentUser.password !== current) return alert('Current password is incorrect.');
    if (!validatePassword(newPass)) return alert('New password must be strong.');
    if (newPass !== confirmPass) return alert('New passwords do not match.');
    const updatedUsers = users.map(u => (u.email === currentUser.email ? { ...u, password: newPass } : u));
    saveUsers(updatedUsers);
    setCurrentUser({ ...currentUser, password: newPass });
    alert('Password changed successfully!');
  };

  const Navbar = () => (
    <div>
      <Sidebar setPage={setPage} />
      <div style={{ marginLeft: '250px', width: 'calc(100% - 250px)' }}>
        <nav style={topNavStyle}>
          <button onClick={() => setPage('dashboard')} style={navButton}>Dashboard</button>
          <button onClick={() => setPage('policies')} style={navButton}>Policies</button>
          <button onClick={() => setPage('settings')} style={navButton}>Settings</button>
          <button onClick={() => { setIsLoggedIn(false); setPage('login'); setCurrentUser(null); }} style={navButton}>Logout</button>
        </nav>
      </div>
    </div>
  );

  const LoginPage = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    return (
      <div style={pageStyle2}>
        <h2>Denial of Service Monitoring System</h2>
        <input placeholder="Email" onChange={e => setEmail(e.target.value)} style={inputStyle} /><br />
        <input type="password" placeholder="Password" onChange={e => setPassword(e.target.value)} style={inputStyle} /><br />
        <button onClick={() => handleLogin(email, password)} style={buttonStyle}>Login</button><br /><br />
        <span>Don't have an account? <button onClick={() => setPage('register')} style={linkButton}>Sign Up</button></span>
      </div>
    );
  };

  const RegisterPage = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirm, setConfirm] = useState('');

    return (
      <div style={pageStyle2}>
        <h2>Create New Account</h2>
        <input placeholder="Email" onChange={e => setEmail(e.target.value)} style={inputStyle} /><br />
        <input type="password" placeholder="Password" onChange={e => setPassword(e.target.value)} style={inputStyle} /><br />
        <input type="password" placeholder="Confirm Password" onChange={e => setConfirm(e.target.value)} style={inputStyle} /><br />
        <button onClick={() => handleRegister(email, password, confirm)} style={buttonStyle}>Register</button><br /><br />
        <button onClick={() => setPage('login')} style={linkButton}>Back to Login</button>
      </div>
    );
  };

  const Dashboard = () => {
    const [logs, setLogs] = useState([]);
    const [currentTime, setCurrentTime] = useState(new Date());
  
    useEffect(() => {
      setLogs([{ time: '2025-04-30 12:34', type: 'SYN Flood' }]);
  
      const interval = setInterval(() => {
        setCurrentTime(new Date());
      }, 1000);
  
      return () => clearInterval(interval);
    }, []);
  
    const handleRefresh = () => {
      window.location.reload();
    };
  
    return (
      <div style={{ ...pageStyle, marginLeft: '260px' }}>
        <h2>Dashboard</h2>
        <div style={cardStyle}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ margin: 0 }}>Attack Logs</h3>
            <div style={{ textAlign: 'right' }}>
              <button
                onClick={handleRefresh}
                style={{
                  ...buttonStyle,
                  width: 'auto',
                  padding: '10px 50px',
                  fontSize: '20px',
                  marginBottom: '5px'
                }}
              >
                Refresh
              </button>
              <div style={{ fontSize: '12px', color: '#aaa' }}>
                {currentTime.toLocaleTimeString()}
              </div>
            </div>
          </div>
          {logs.map((log, i) => (
            <p key={i}>{log.time} - {log.type}</p>
          ))}
        </div>
  
        <div style={{ ...cardStyle, marginBottom: 20 }}>
          <h3>Detected DoS Attacks:</h3>
          <p>Last Attack:</p>
          <img
            src="/last-attack.png"
            alt="Last DoS Attack Graph"
            style={{ width: '100%', maxHeight: '300px', objectFit: 'contain', borderRadius: '8px' }}
          />
        </div>
  
        <div style={{ ...cardStyle, marginBottom: 20 }}>
          <h3>Current System Status:</h3>
          <p style={{ color: 'lime' }}>Operational</p>
        </div>
      </div>
    );
  };
  

  const Reports = () => (
    <div style={{ ...pageStyle, marginLeft: '260px' }}>
      <h2>Reports</h2>
      <p>This is the Reports page content.</p>
    </div>
  );

  const Policies = () => (
    <div style={{ ...pageStyle, marginLeft: '260px' }}>
      <h2>Policy Management</h2>
      <div style={cardStyle}>
        <h3>Last Policy Update:</h3>
        <p>Status: <span style={{ color: 'lime' }}>Active</span></p>
        <button style={buttonStyle}>Update Policies Now</button>
      </div>
    </div>
  );

  const Settings = () => {
    const [current, setCurrent] = useState('');
    const [newPass, setNewPass] = useState('');
    const [confirmPass, setConfirmPass] = useState('');
    const [newEmail, setNewEmail] = useState(currentUser?.email || '');
    const [newUsername, setNewUsername] = useState(currentUser?.username || '');

    const handleUpdateProfile = () => {
      if (currentUser.email === 'admin') return alert('Admin profile cannot be updated.');
      if (!current || currentUser.password !== current) return alert('Current password is incorrect.');
      if (!validateEmail(newEmail)) return alert('Invalid email.');
      const emailTaken = users.some(u => u.email === newEmail && u.email !== currentUser.email);
      if (emailTaken) return alert('Email is already taken.');
      const updatedUsers = users.map(u => u.email === currentUser.email ? { ...u, email: newEmail, username: newUsername } : u);
      saveUsers(updatedUsers);
      setCurrentUser({ ...currentUser, email: newEmail, username: newUsername });
      alert('Profile updated successfully!');
    };

    return (
      <div style={{ ...pageStyle, marginLeft: '260px' }}>
        <h2>Settings</h2>
        <div style={{ ...cardStyle, maxWidth: '500px' }}>
          <h3>Account: {currentUser?.email}</h3>
          <input type="text" placeholder="New Username" value={newUsername} onChange={e => setNewUsername(e.target.value)} style={inputStyle} /><br />
          <input type="email" placeholder="New Email" value={newEmail} onChange={e => setNewEmail(e.target.value)} style={inputStyle} /><br />
          <input type="password" placeholder="Current Password (required)" onChange={e => setCurrent(e.target.value)} style={inputStyle} /><br />
          <button onClick={handleUpdateProfile} style={buttonStyle}>Update Profile</button>
        </div>
        <div style={{ ...cardStyle, marginTop: 30, maxWidth: '500px' }}>
          <h3>Change Password</h3>
          <input type="password" placeholder="Current Password" onChange={e => setCurrent(e.target.value)} style={inputStyle} /><br />
          <input type="password" placeholder="New Password" onChange={e => setNewPass(e.target.value)} style={inputStyle} /><br />
          <input type="password" placeholder="Confirm New Password" onChange={e => setConfirmPass(e.target.value)} style={inputStyle} /><br />
          <button onClick={() => handleChangePassword(current, newPass, confirmPass)} style={buttonStyle}>Change Password</button>
        </div>
      </div>
    );
  };

  const Whitelist = () => (<div style={{ ...pageStyle, marginLeft: '260px' }}><h2>Whitelist Page</h2><p>This is the Whitelist content.</p></div>);
  const Blacklist = () => (<div style={{ ...pageStyle, marginLeft: '260px' }}><h2>Blacklist Page</h2><p>This is the Blacklist content.</p></div>);
  const Sample1 = () => (<div style={{ ...pageStyle, marginLeft: '260px' }}><h2>Sample1 Page</h2><p>This is the Sample1 content.</p></div>);
  const Sample11 = () => (<div style={{ ...pageStyle, marginLeft: '260px' }}><h2>Sample11 Page</h2><p>Sample11</p></div>);

  return (
    <div style={{ backgroundColor: '#121212', minHeight: '100vh', color: '#fff' }}>
      {isLoggedIn && <Navbar />}
      {!isLoggedIn && page === 'login' && <LoginPage />}
      {!isLoggedIn && page === 'register' && <RegisterPage />}
      {isLoggedIn && page === 'dashboard' && <Dashboard />}
      {isLoggedIn && page === 'policies' && <Policies />}
      {isLoggedIn && page === 'settings' && <Settings />}
      {isLoggedIn && page === 'whitelist' && <Whitelist />}
      {isLoggedIn && page === 'blacklist' && <Blacklist />}
      {isLoggedIn && page === 'Sample1' && <Sample1 />}
      {isLoggedIn && page === 'Sample11' && <Sample11 />}
      {isLoggedIn && page === 'reports' && <Reports />}
    </div>
  );
}

const sidebarIconContainer = { display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' };
const iconStyle = { width: '50px', maxWidth: '50px', objectFit: 'contain', marginBottom: '5px' };
const sidebarStyle = { position: 'fixed', top: 0, left: 0, width: '200px', height: '100%', backgroundColor: '#1e1e1e', padding: '20px 10px', overflowY: 'auto', borderRadius: '0 20px 20px 0' };
const sidebarHeader = { textAlign: 'center', marginBottom: 10, color: '#fff' };
const sidebarList = { listStyleType: 'none', padding: 0 };
const sidebarItem = { padding: '10px', color: '#ccc', cursor: 'pointer' };
const topNavStyle = { backgroundColor: '#1f1f1f', padding: 10, display: 'flex', justifyContent: 'center', borderRadius: '0 0 12px 12px' };
const navButton = { color: '#fff', background: 'none', border: 'none', marginRight: 20, fontSize: '16px', cursor: 'pointer' };
const pageStyle = { padding: 20, fontFamily: 'Arial, sans-serif' };
const pageStyle2 = { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', fontFamily: 'Arial, sans-serif', padding: 20, textAlign: 'center' };
const cardStyle = { backgroundColor: '#1e1e1e', padding: 20, marginTop: 20, borderRadius: 16, boxShadow: '0 0 10px rgba(0,0,0,0.5)' };
const inputStyle = {
  padding: 10,
  margin: '10px 0',
  borderRadius: 8,
  border: '1px solid #ccc',
  width: '100%',
  maxWidth: '300px', // 👈 Limit input box width
  backgroundColor: '#1a1a1a',
  color: '#fff'
};

const buttonStyle = { padding: 10,maxWidth: '400px', backgroundColor: '#333', color: '#fff', borderRadius: 8, border: 'none', cursor: 'pointer', width: '100%' };
const linkButton = { color: '#00f', background: 'none', border: 'none', textDecoration: 'underline', cursor: 'pointer' };

export default App;
