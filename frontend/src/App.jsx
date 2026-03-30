import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import './index.css';

const API = '';

function FormattedText({ text, onPageClick }) {
  if (!text) return null;

  const hasSteps = /(?:^|\n)\s*\d+\.\s/.test(text);

  if (hasSteps) {
    const parts = text.split(/(?:^|\n)(?=\d+\.\s)/);
    const intro = parts[0] && !/^\d+\.\s/.test(parts[0].trim()) ? parts.shift() : null;

    return (
      <div className="formatted-text">
        {intro && <p className="response-intro">{renderInline(intro.trim(), onPageClick)}</p>}
        <ol className="response-steps">
          {parts.map((step, i) => {
            const cleaned = step.replace(/^\d+\.\s*/, '').trim();
            if (!cleaned) return null;
            return <li key={i}>{renderInline(cleaned, onPageClick)}</li>;
          })}
        </ol>
      </div>
    );
  }

  // No steps — render as paragraph(s)
  const paragraphs = text.split(/\n{2,}/);
  return (
    <div className="formatted-text">
      {paragraphs.map((p, i) => (
        <p key={i} className="response-para">{renderInline(p.trim(), onPageClick)}</p>
      ))}
    </div>
  );
}

/** Render inline elements: page refs as bubbles, **bold**, and plain text */
function renderInline(text, onPageClick) {
  // Pattern: (Page N), page N, Page N, or [Page N]
  const parts = text.split(/(\(?[Pp]age\s+\d+\)?|\[[Pp]age\s+\d+\])/);
  return parts.map((part, i) => {
    // Match page reference
    const pageMatch = part.match(/[Pp]age\s+(\d+)/);
    if (pageMatch) {
      const pageNum = parseInt(pageMatch[1]);
      return (
        <span key={i} className="page-ref clickable" data-tooltip={`Source: Ford F-150 Workshop Manual — Page ${pageNum}`}
              onClick={(e) => { e.stopPropagation(); onPageClick && onPageClick(pageNum); }}>
          <span className="page-ref-icon">📄</span> p.{pageNum}
        </span>
      );
    }
    // Render bold markers **text**
    return renderBold(part, i);
  });
}

function renderBold(text, keyPrefix) {
  const parts = text.split(/(\*\*[^*]+\*\*)/);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={`${keyPrefix}-b${i}`}>{part.slice(2, -2)}</strong>;
    }
    return <span key={`${keyPrefix}-t${i}`}>{part}</span>;
  });
}

function TypewriterText({ text, speed = 12, onPageClick }) {
  const [displayed, setDisplayed] = useState('');
  const [done, setDone] = useState(false);

  useEffect(() => {
    setDisplayed('');
    setDone(false);
    let i = 0;
    const timer = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(timer);
        setDone(true);
      }
    }, speed);
    return () => clearInterval(timer);
  }, [text, speed]);

  if (done) return <FormattedText text={text} onPageClick={onPageClick} />;
  return <FormattedText text={displayed + '▌'} onPageClick={onPageClick} />;
}

function App() {
  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [renamingId, setRenamingId] = useState(null);
  const [animatingMsgId, setAnimatingMsgId] = useState(null);
  const [pdfPage, setPdfPage] = useState(null);
  const [sidebarW, setSidebarW] = useState(240);
  const [pdfW, setPdfW] = useState(420);
  const [renameVal, setRenameVal] = useState('');
  const [emojiPicker, setEmojiPicker] = useState(null);
  const [navRenaming, setNavRenaming] = useState(false);
  const [navRenameVal, setNavRenameVal] = useState('');
  const endRef = useRef(null);
  const dragging = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  const onMouseDown = useCallback((panel) => (e) => {
    e.preventDefault();
    dragging.current = panel;
    setIsDragging(true);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  useEffect(() => {
    const onMove = (e) => {
      if (!dragging.current) return;
      if (dragging.current === 'sidebar') {
        const w = Math.max(160, Math.min(400, e.clientX));
        setSidebarW(w);
      } else if (dragging.current === 'pdf') {
        const w = Math.max(280, Math.min(800, window.innerWidth - e.clientX));
        setPdfW(w);
      }
    };
    const onUp = () => {
      dragging.current = null;
      setIsDragging(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, []);

  // Close emoji picker on any outside click
  useEffect(() => {
    if (!emojiPicker) return;
    const close = () => setEmojiPicker(null);
    window.addEventListener('click', close);
    return () => window.removeEventListener('click', close);
  }, [emojiPicker]);

  useEffect(() => {
    axios.get(`${API}/api/chats`).then(r => {
      const c = r.data.chats || [];
      setChats(c);
      if (c.length > 0) setActiveChat(c[0].id);
    }).catch(() => {});
  }, []);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); });

  const current = chats.find(c => c.id === activeChat);
  const messages = current?.messages || [];

  const save = (data) => axios.post(`${API}/api/chats`, { chats: data }).catch(() => {});

  const [cacheCleared, setCacheCleared] = useState(false);
  const clearCache = async () => {
    await axios.delete(`${API}/api/cache`).catch(() => {});
    setCacheCleared(true);
    setTimeout(() => setCacheCleared(false), 2000);
  };

  const newChat = () => {
    const id = Date.now().toString();
    const chat = { id, title: 'New chat', messages: [] };
    const updated = [chat, ...chats];
    setChats(updated);
    setActiveChat(id);
    save(updated);
  };

  const deleteChat = (id) => {
    const updated = chats.filter(c => c.id !== id);
    setChats(updated);
    if (activeChat === id) setActiveChat(updated[0]?.id || null);
    axios.delete(`${API}/api/chats/${id}`).catch(() => {});
  };

  const startRename = (id, title) => { setRenamingId(id); setRenameVal(title); };
  const finishRename = () => {
    if (!renamingId) return;
    const updated = chats.map(c => c.id === renamingId ? { ...c, title: renameVal } : c);
    setChats(updated);
    save(updated);
    setRenamingId(null);
  };

  const getUserMessages = () => {
    return messages.filter(m => m.role === 'user').map(m => m.content).slice(-3);
  };

  const autoTitleChat = async (chatId, question, answer) => {
    try {
      const r = await axios.post(`${API}/api/chats/${chatId}/autotitle`, { question, answer });
      const title = r.data.title;
      if (title) {
        setChats(prev => {
          const next = prev.map(c => c.id === chatId ? { ...c, title } : c);
          save(next);
          return next;
        });
      }
    } catch {}
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const isFirstMessage = current?.messages?.length === 0;
    const userMsg = { role: 'user', content: input, ts: Date.now() };
    const inputSnapshot = input;
    let updated = chats.map(c =>
      c.id === activeChat ? { ...c, messages: [...c.messages, userMsg] } : c
    );
    setChats(updated);
    setInput('');
    setLoading(true);

    try {
      const res = await axios.post(`${API}/api/query`, {
        query: input,
        top_k: 15,
        chat_context: getUserMessages(),
        chat_id: activeChat
      });
      const specs = res.data.specs || [];
      const cached = res.data.cached || false;
      const source = res.data.source || 'rag';
      const answer = res.data.answer || '';

      if (cached || source === 'cache' || source === 'session' || source === 'spec_db') {
        await new Promise(r => setTimeout(r, 2000));
      }

      let content;
      if (specs.length === 0 && answer) {
        content = answer;
      } else if (specs.length === 0) {
        content = 'No specifications found for this query.';
      } else {
        content = { specs, answer };
      }

      const msgId = Date.now().toString();
      const assistantMsg = {
        role: 'assistant',
        content,
        cached,
        source,
        ts: Date.now(),
        msgId
      };

      setAnimatingMsgId(msgId);

      updated = updated.map(c => {
        if (c.id === activeChat) {
          return { ...c, messages: [...c.messages, assistantMsg] };
        }
        return c;
      });
      setChats(updated);
      save(updated);

      if (isFirstMessage) {
        autoTitleChat(activeChat, inputSnapshot, typeof answer === 'string' ? answer : '');
      }
    } catch (e) {
      console.error('API Error:', e);
      const errMsg = { role: 'assistant', content: `Error: Could not reach API. Details: ${e.message}`, ts: Date.now() };
      updated = updated.map(c => c.id === activeChat ? { ...c, messages: [...c.messages, errMsg] } : c);
      setChats(updated);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`layout${pdfPage ? ' pdf-open' : ''}`}>
      {isDragging && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 9998, cursor: 'col-resize' }} />
      )}
      <aside className="sidebar" style={{ width: sidebarW }}>
        <div className="sidebar-title">Vehicle Spec Extraction</div>
        <button className="new-btn" onClick={newChat}>+ New Chat</button>
        <button className="clear-cache-btn" onClick={clearCache}>
          {cacheCleared ? '✓ Cleared!' : '🗑 Clear Cache'}
        </button>
        <div className="chat-list">
          {chats.map(c => (
            <div key={c.id} className={`chat-item ${c.id === activeChat ? 'active' : ''}`} onClick={() => { setActiveChat(c.id); setEmojiPicker(null); }}>
              {renamingId === c.id ? (
                <input
                  className="rename-input"
                  value={renameVal}
                  onChange={e => setRenameVal(e.target.value)}
                  onBlur={finishRename}
                  onKeyDown={e => e.key === 'Enter' && finishRename()}
                  autoFocus
                  onClick={e => e.stopPropagation()}
                />
              ) : (
                <>
                  <span className="chat-item-emoji" title="Change emoji"
                    onClick={e => {
                      e.stopPropagation();
                      if (emojiPicker?.id === c.id) { setEmojiPicker(null); return; }
                      const rect = e.currentTarget.getBoundingClientRect();
                      setEmojiPicker({ id: c.id, x: rect.right + 6, y: rect.top });
                    }}>
                    {c.emoji || '💬'}
                  </span>
                  <span className="chat-item-title">{c.title}</span>
                  <span className="chat-item-actions">
                    <button className="chat-action-btn" onClick={e => { e.stopPropagation(); startRename(c.id, c.title); }} title="Rename">✎</button>
                    <button className="chat-action-btn" onClick={e => { e.stopPropagation(); deleteChat(c.id); }} title="Delete">✕</button>
                  </span>
                </>
              )}
            </div>
          ))}
        </div>
      </aside>

      <div className="resize-handle" onMouseDown={onMouseDown('sidebar')} />

      <main className="main">
        {!activeChat ? (
          <div className="empty">Start a new chat</div>
        ) : (
        <>
            <div className="chat-navbar">
              {current && (
                navRenaming ? (
                  <input
                    className="nav-rename-input"
                    value={navRenameVal}
                    autoFocus
                    onChange={e => setNavRenameVal(e.target.value)}
                    onBlur={() => {
                      const trimmed = navRenameVal.trim();
                      if (trimmed) {
                        const next = chats.map(c => c.id === activeChat ? { ...c, title: trimmed } : c);
                        setChats(next);
                        save(next);
                      }
                      setNavRenaming(false);
                    }}
                    onKeyDown={e => {
                      if (e.key === 'Enter') e.target.blur();
                      if (e.key === 'Escape') setNavRenaming(false);
                    }}
                  />
                ) : (
                  <span className="nav-title" onClick={() => { setNavRenameVal(current.title); setNavRenaming(true); }} title="Click to rename">
                    {current.emoji || '💬'} {current.title}
                  </span>
                )
              )}
              {!navRenaming && (
                <button className="nav-rename-btn" title="Rename chat" onClick={() => { setNavRenameVal(current?.title || ''); setNavRenaming(true); }}>✎</button>
              )}
            </div>
            <div className="messages">
              {messages.map((msg, idx) => {
                const isNew = msg.msgId && msg.msgId === animatingMsgId;
                return (
                <div key={idx} className={`msg ${msg.role}${isNew ? ' msg-animate' : ''}`}>
                  <div className="msg-label">{msg.role === 'user' ? 'You' : 'System'}</div>
                  {typeof msg.content === 'string' ? (
                    <div className="msg-text">
                      {isNew ? <TypewriterText text={msg.content} onPageClick={setPdfPage} /> : <FormattedText text={msg.content} onPageClick={setPdfPage} />}
                    </div>
                  ) : Array.isArray(msg.content) ? (
                    <SpecResult specs={msg.content} answer="" cached={msg.cached} source={msg.source} isNew={isNew} onPageClick={setPdfPage} />
                  ) : (
                    <SpecResult specs={msg.content.specs} answer={msg.content.answer} showTable={msg.content.show_table !== false} cached={msg.cached} source={msg.source} isNew={isNew} onPageClick={setPdfPage} />
                  )}
                </div>
                );
              })}
              {loading && (
                <div className="msg assistant">
                  <div className="msg-label">System</div>
                  <div className="dots"><span /><span /><span /></div>
                </div>
              )}
              <div ref={endRef} />
            </div>
            <div className="input-bar">
              <form className="input-form" onSubmit={handleSend}>
                <input className="input-field" placeholder="Ask about specs..." value={input} onChange={e => setInput(e.target.value)} />
                <button type="submit" className="send-btn" disabled={loading}>Send</button>
              </form>
            </div>
          </>
        )}
      </main>

      {emojiPicker && (
        <div className="emoji-popup" style={{ position: 'fixed', left: emojiPicker.x, top: emojiPicker.y, zIndex: 9999 }}
             onClick={e => e.stopPropagation()}>
          {['💬','🔧','🚗','⚙️','🔩','🛞','🛻','📋','📌','🔍','💡','🏎️','🛠️','📊','🔋','🔌','🧰','🛡️','🔬','🌡️'].map(em => (
            <span key={em} className="emoji-option" onClick={() => {
              const next = chats.map(ch => ch.id === emojiPicker.id ? { ...ch, emoji: em } : ch);
              setChats(next);
              save(next);
              setEmojiPicker(null);
            }}>{em}</span>
          ))}
        </div>
      )}

      {pdfPage && (
        <>
          <div className="resize-handle" onMouseDown={onMouseDown('pdf')} />
          <aside className="pdf-panel" style={{ width: pdfW }}>
            <div className="pdf-panel-header">
              <span className="pdf-panel-title">📄 Manual — Page {pdfPage}</span>
              <button className="pdf-panel-close" onClick={() => setPdfPage(null)}>✕</button>
            </div>
            <iframe
              key={pdfPage}
              className="pdf-iframe"
              src={`/manual.pdf#page=${pdfPage}`}
              title="PDF Preview"
            />
          </aside>
        </>
      )}
    </div>
  );
}

function SpecResult({ specs = [], answer, showTable = true, cached, source, isNew = false, onPageClick }) {
  const [filter, setFilter] = useState('');
  const [showFilter, setShowFilter] = useState(false);

  const sourceLabels = { cache: 'cached', session: 'from session', spec_db: 'instant lookup', rag: 'live extraction' };

  const filtered = filter
    ? specs.filter(s =>
        s.component.toLowerCase().includes(filter.toLowerCase()) ||
        s.spec_type.toLowerCase().includes(filter.toLowerCase()) ||
        s.value.includes(filter)
      )
    : specs;

  const exportCSV = () => {
    const header = 'Component,Type,Value,Unit,Page\n';
    const rows = filtered.map(s => `"${s.component}","${s.spec_type}","${s.value}","${s.unit}",${s.source_page}`).join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'specifications.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      {answer && <div className="msg-text" style={{ marginBottom: '0.75rem' }}>
        {isNew ? <TypewriterText text={answer} onPageClick={onPageClick} /> : <FormattedText text={answer} onPageClick={onPageClick} />}
      </div>}
      {showTable && filtered.length > 0 && (
        <>
          <div className="spec-header">
            <span className="spec-count">
              {filtered.length} spec{filtered.length !== 1 ? 's' : ''}
              {source && <span style={{opacity:0.5}}> · {sourceLabels[source] || source}</span>}
            </span>
            <div className="spec-tools">
              {showFilter && (
                <input
                  className="filter-input"
                  placeholder="Filter components..."
                  value={filter}
                  onChange={e => setFilter(e.target.value)}
                  autoFocus
                />
              )}
              <button className="tool-btn" onClick={() => setShowFilter(!showFilter)}>
                {showFilter ? '✕' : 'Filter'}
              </button>
              <button className="tool-btn" onClick={exportCSV}>Export CSV</button>
            </div>
          </div>
          <table className="spec-table">
            <thead>
              <tr><th>Component</th><th>Type</th><th>Value</th><th>Page</th></tr>
            </thead>
            <tbody>
              {filtered.map((s, i) => (
                <tr key={i}>
                  <td>{s.component}</td>
                  <td>{s.spec_type}</td>
                  <td className="val">{s.value} {s.unit}</td>
                  <td>
                    <span className="page-ref clickable" data-tooltip={`Source: Ford F-150 Workshop Manual — Page ${s.source_page}`}
                          onClick={() => onPageClick && onPageClick(s.source_page)}>
                      <span className="page-ref-icon">📄</span> p.{s.source_page}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

export default App;
