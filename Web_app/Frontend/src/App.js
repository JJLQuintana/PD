import React, { useState, useEffect } from 'react';
import Systemconfig from './Systemconfig'



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
        <li style={sidebarItem} onClick={() => setPage('system configuration')}>
          <div style={sidebarIconContainer}>
            <span>Systemconfig</span>
          </div>
        </li>
        <li style={sidebarItem} onClick={() => setPage('authorized admin')}>
          <div style={sidebarIconContainer}>
            <span>Admin</span>
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
      setPage('detection logs');
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
      setPage('detection logs');
      return;
    }

    const user = users.find(u => u.email === email && u.password === password);
    if (user) {
      setCurrentUser(user);
      setIsLoggedIn(true);
      setPage('detection logs');
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
          <button onClick={() => setPage('detection logs')} style={navButton}>Detection Logs</button>
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
        <div style={{ textAlign: 'center' }}>
          <img
           src="/logofront.png" 
           alt="DeepShield Logo" 
           style={{ width: '250px', marginBottom: '5px' }} 
        />
          <h2 style = {{ marginTop: '0' }}> Denial of Service Monitoring System</h2>
</div>

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
  
  const DetectionLogs = () => {
  const [logs, setLogs] = useState([]);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => {
      fetchPrediction();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchPrediction = async () => {
  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const data = await response.json(); // data is an array

    if (!Array.isArray(data)) {
      console.error('Unexpected response format:', data);
      return;
    }

    const formattedLogs = data.map(entry => ({
      time: new Date().toLocaleString(),
      type: entry.label, // Use label from backend
      policy: `Confidence: ${entry.confidence.toFixed(6)}`
    }));

    setLogs(prevLogs => [...formattedLogs, ...prevLogs]);
  } catch (error) {
    console.error('Error fetching prediction:', error);
  }
};




  return (
    <div style={{ ...pageStyle, marginLeft: '260px' }}>
      <h2>Detection Logs</h2>
      <div style={cardStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0 }}>Attack Logs</h3>
          <div style={{ textAlign: 'right' }}>
            <button onClick={() => setCurrentTime(new Date())} style={{ ...buttonStyle, width: 'auto', padding: '10px 50px', fontSize: '20px', marginBottom: '5px' }}>
              Refresh
            </button>
            <div style={{ fontSize: '12px', color: '#aaa' }}>{currentTime.toLocaleTimeString()}</div>
          </div>
        </div>
        <button onClick={fetchPrediction} style={buttonStyle}>Run Detection</button>
        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '20px' }}>
          <thead>
            <tr>
              <th style={tableHeaderStyle}>Timestamp</th>
              <th style={tableHeaderStyle}>Classification</th>
              <th style={tableHeaderStyle}>Details</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log, index) => (
              <tr key={index}>
                <td style={tableCellStyle}>{log.time}</td>
                <td style={tableCellStyle}>{log.type}</td>
                <td style={tableCellStyle}>{log.policy}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

  
  
  
  
  // Table styling
  const tableHeaderStyle = {
    borderBottom: '2px solid #ccc',
    padding: '10px',
    textAlign: 'left',
    backgroundColor: '#212224',
  };
  
  const tableCellStyle = {
    padding: '10px',
    borderBottom: '1px solid #eee',
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

  const Whitelist = () => {
    const [whitelist, setWhitelist] = useState(() => {
      return JSON.parse(localStorage.getItem('dosWhitelist')) || [];
    });
    const [ipInput, setIpInput] = useState('');
  
    const saveWhitelist = (list) => {
      localStorage.setItem('dosWhitelist', JSON.stringify(list));
      setWhitelist(list);
    };
  
    const handleAddIp = () => {
      const trimmed = ipInput.trim();
      const ipRegex = /^(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}$/;
  
      if (!ipRegex.test(trimmed)) {
        alert('Invalid IP address.');
        return;
      }
      if (whitelist.includes(trimmed)) {
        alert('IP already whitelisted.');
        return;
      }
  
      saveWhitelist([...whitelist, trimmed]);
      setIpInput('');
    };
  
    const handleRemoveIp = (ip) => {
      const updatedList = whitelist.filter(item => item !== ip);
      saveWhitelist(updatedList);
    };
  
    return (
      <div style={{ ...pageStyle, marginLeft: '260px' }}>
        <h2>Whitelist Page</h2>
  
        <div style={cardStyle}>
          <h3>Trusted IP Addresses</h3>
  
          <input
            type="text"
            value={ipInput}
            onChange={e => setIpInput(e.target.value)}
            placeholder="Enter IP address"
            style={inputStyle}
          />
          <button onClick={handleAddIp} style={buttonStyle}>Add IP</button>
  
          <ul style={{ marginTop: 20 }}>
            {whitelist.length === 0 && <p>No IPs whitelisted.</p>}
            {whitelist.map(ip => (
              <li key={ip} style={{ marginBottom: 10 }}>
                {ip}
                <button
                  onClick={() => handleRemoveIp(ip)}
                  style={{ ...buttonStyle, marginLeft: 10, backgroundColor: '#822' }}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  };
  
  const Blacklist = () => {
    const [blacklist, setBlacklist] = useState(() => {
      return JSON.parse(localStorage.getItem('dosBlacklist')) || [];
    });
    const [ipInput, setIpInput] = useState('');
  
    const saveBlacklist = (list) => {
      localStorage.setItem('dosBlacklist', JSON.stringify(list));
      setBlacklist(list);
    };
  
    const handleAddIp = () => {
      const trimmed = ipInput.trim();
      const ipRegex = /^(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}$/;
  
      if (!ipRegex.test(trimmed)) {
        alert('Invalid IP address.');
        return;
      }
      if (blacklist.includes(trimmed)) {
        alert('IP already blacklisted.');
        return;
      }
  
      saveBlacklist([...blacklist, trimmed]);
      setIpInput('');
    };
  
    const handleRemoveIp = (ip) => {
      const updatedList = blacklist.filter(item => item !== ip);
      saveBlacklist(updatedList);
    };
  
    return (
      <div style={{ ...pageStyle, marginLeft: '260px' }}>
        <h2>Blacklist Page</h2>
  
        <div style={cardStyle}>
          <h3>Blocked IP Addresses</h3>
  
          <input
            type="text"
            value={ipInput}
            onChange={e => setIpInput(e.target.value)}
            placeholder="Enter IP address"
            style={inputStyle}
          />
          <button onClick={handleAddIp} style={buttonStyle}>Add IP</button>
  
          <ul style={{ marginTop: 20 }}>
            {blacklist.length === 0 && <p>No IPs blacklisted.</p>}
            {blacklist.map(ip => (
              <li key={ip} style={{ marginBottom: 10 }}>
                {ip}
                <button
                  onClick={() => handleRemoveIp(ip)}
                  style={{ ...buttonStyle, marginLeft: 10, backgroundColor: '#a00' }}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  };
  
  const SystemConfig = () => (<div style={{ ...pageStyle, marginLeft: '260px' }}><h2>SystemConfig Page</h2><p>This is the Sample1 content.</p></div>);
  const Admin = () => (<div style={{ ...pageStyle, marginLeft: '260px' }}><h2>Authorized Admin</h2><p>Admin</p></div>);
  

  return (
    <div style={{ backgroundColor: '#121212', minHeight: '100vh', color: '#fff' }}>
      {isLoggedIn && <Navbar />}
      {!isLoggedIn && page === 'login' && <LoginPage />}
      {!isLoggedIn && page === 'register' && <RegisterPage />}
      {isLoggedIn && page === 'detection logs' && <DetectionLogs />}
      {isLoggedIn && page === 'policies' && <Policies />}
      {isLoggedIn && page === 'settings' && <Settings />}
      {isLoggedIn && page === 'whitelist' && <Whitelist />}
      {isLoggedIn && page === 'blacklist' && <Blacklist />}
      {isLoggedIn && page === 'systemconfig' && <SystemConfig />}
      {isLoggedIn && page === 'authorized admin' && <Admin />}
      {isLoggedIn && page === 'reports' && <Reports />}
      {isLoggedIn && page === 'system configuration' && <Systemconfig />}
      
      
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